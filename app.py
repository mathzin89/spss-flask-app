from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
import pyreadstat
from uuid import uuid4
import io, os, tempfile, traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "dev-secret"  # troque em produção

# token -> {"df": df, "meta": meta, "name": filename, "last_params": {...}, "last_tables": [...]}
DATASETS = {}


# =========================
# Leitura do arquivo
# =========================
def read_dataset(file_storage):
    name = secure_filename(file_storage.filename or "")
    ext = os.path.splitext(name)[1].lower()
    data = file_storage.read()
    try:
        file_storage.seek(0)
    except Exception:
        pass

    if ext in (".sav", ".zsav"):
        # pyreadstat exige path; gravamos temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            df, meta = pyreadstat.read_sav(tmp_path, apply_value_formats=False)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return df, meta, name

    elif ext == ".csv":
        df = pd.read_csv(io.BytesIO(data))
        meta = None
        return df, meta, name

    else:
        raise ValueError("Formato não suportado. Envie .sav, .zsav ou .csv")


# =========================
# Helpers de labels e pesos
# =========================
def _weight_series(df, weight_var):
    if weight_var and weight_var in df.columns:
        w = pd.to_numeric(df[weight_var], errors="coerce").fillna(0.0)
    else:
        w = pd.Series(1.0, index=df.index)
    w.index = df.index
    return w


def _value_labels(meta, var):
    """
    Retorna {codigo: rotulo} ou None.

    Formatos possíveis no pyreadstat:
      1) meta.value_labels[var] = {code: label}
      2) meta.variable_value_labels[var] = "labelset"  -> usar meta.value_labels["labelset"]
      3) meta.variable_value_labels[var] = {code: label}  -> já é o mapa final
    """
    if meta is None or var is None:
        return None

    # Normaliza caso venha objeto/dict do front
    if not isinstance(var, (str, int)):
        try:
            var = var.get("value") or var.get("name") or str(var)
        except Exception:
            var = str(var)

    vvl = getattr(meta, "value_labels", None) or {}

    # Caso 1: rótulo diretamente por variável
    direct = vvl.get(var)
    if isinstance(direct, dict):
        return direct

    # Mapeamento var -> labelset
    var_to_labelset = getattr(meta, "variable_value_labels", None) or {}
    labelset = var_to_labelset.get(var)

    # Caso 3: já é um dict {codigo:rotulo}
    if isinstance(labelset, dict):
        return labelset

    # Caso 2: labelset é chave (hashable) que aponta para dict em value_labels
    if isinstance(labelset, (str, int)):
        maybe = vvl.get(labelset)
        if isinstance(maybe, dict):
            return maybe

    return None


def _var_label(meta, var):
    if meta is None or var is None:
        return var or ""
    try:
        names = getattr(meta, "column_names", []) or []
        labs = getattr(meta, "column_labels", []) or []
        mapping = {n: l for n, l in zip(names, labs) if l}
        return mapping.get(var, var)
    except Exception:
        return var


def _fmt_pct(x, dec=1):
    if pd.isna(x):
        return ""
    # vírgula decimal; sem símbolo %
    s = f"{x:.{dec}f}".replace(".", ",")
    return s


# ===========================================
# Renderer "simples" (legado) – ainda disponível
# ===========================================
def render_table_html(table_dict):
    thead_cols = "".join(f"<th>{c}</th>" for c in table_dict["columns"])
    rows_html = "".join(
        "<tr><td class='rowlbl'>{}</td>{}</tr>".format(
            rlbl, "".join(f"<td class='cell'>{c}</td>" for c in cells)
        )
        for (rlbl, cells) in table_dict["rows"]
    )
    base_html = "<tr class='base'><td class='rowlbl'>Base</td>{}</tr>".format(
        "".join(f"<td class='cell'>{b}</td>" for b in table_dict["base"])
    )
    return f"""
      <div class="tbl">
        <div class="title">{table_dict['title']}</div>
        <div class="caption">Pergunta: {table_dict['caption'].replace('Pergunta: ', '')}</div>
        <table class="ctab">
          <thead><tr><th></th>{thead_cols}</tr></thead>
          <tbody>
            {rows_html}
            {base_html}
          </tbody>
        </table>
      </div>
    """


# ===========================================================
# TABELA LARGA: UMA por alvo (Total + TODOS os BYs em colunas)
# ===========================================================
def compute_table_wide(
    df, target, by_vars=None, weight=None,
    percent_axis="col", base_weighted=True, decimals=1, meta=None
):
    # Normalização defensiva
    if target is not None and not isinstance(target, (str, int)):
        try:
            target = target.get("value") or target.get("name") or str(target)
        except Exception:
            target = str(target)

    by_vars = by_vars or []
    by_vars = [
        (b.get("value") or b.get("name") if isinstance(b, dict) else b)
        for b in by_vars
    ]
    by_vars = [str(b) for b in by_vars if b not in (None, "")]

    w_all = _weight_series(df, weight)

    # Linhas (alvo)
    targ_vmap = _value_labels(meta, target) or {}
    if targ_vmap:
        targ_codes = list(targ_vmap.keys())
        targ_row_labels = [targ_vmap[c] for c in targ_codes]
    else:
        targ_codes = list(pd.unique(df[target].dropna()))
        targ_row_labels = [str(c) for c in targ_codes]

    # Colunas: começa com Total
    columns = [("", "Total")]                                   # (grupo, rótulo)
    col_masks = [pd.Series(True, index=df.index)]               # máscara por coluna (mesmo índice da base)
    # Para cada BY, adiciona colunas por categoria
    for by in by_vars:
        vmap = _value_labels(meta, by) or {}
        if vmap:
            codes = list(vmap.keys())
            labels = [vmap[c] for c in codes]
        else:
            codes = list(pd.unique(df[by].dropna()))
            labels = [str(c) for c in codes]
        grp_name = _var_label(meta, by) or by
        for code, lab in zip(codes, labels):
            columns.append((grp_name, lab))
            col_masks.append(df[by] == code)

    # Denominadores por coluna (soma de pesos)
    denoms = [(w_all[m].sum() if percent_axis == "col" else None) for m in col_masks]

    # Linhas da tabela
    rows = []
    for r_code, r_label in zip(targ_codes, targ_row_labels):
        row_cells = []
        row_mask = (df[target] == r_code)  # alinhado à base completa
        for j, mcol in enumerate(col_masks):
            num = w_all[row_mask & mcol].sum()
            if percent_axis == "col":
                denom = denoms[j]
            elif percent_axis == "row":
                denom = w_all[row_mask].sum()
            else:
                denom = w_all.sum()
            pct = (num / denom * 100.0) if denom and denom > 0 else np.nan
            row_cells.append(_fmt_pct(pct, decimals))
        rows.append((r_label, row_cells))

    # Base por coluna
    base_cells = []
    for mcol in col_masks:
        base_val = int(round(w_all[mcol].sum())) if base_weighted else int(mcol.sum())
        base_cells.append(str(base_val))

    return {
        "title": _var_label(meta, target),
        "caption": f"Pergunta: {_var_label(meta, target)}",
        "columns": [c[1] for c in columns],  # para o renderer simples (uma linha de header)
        "rows": rows,
        "base": base_cells,
        # Para export (quando quisermos 2 níveis de header)
        "_columns_grouped": columns
    }


def render_table_html_wide(table_dict):
    """
    Renderiza a tabela larga com 2 linhas de cabeçalho: grupos (colspan) e rótulos.
    Mantém visual parecido ao seu template atual.
    """
    columns = table_dict.get("_columns_grouped") or [("", c) for c in table_dict["columns"]]
    rows = table_dict["rows"]
    base_cells = table_dict["base"]

    # colspans da linha de grupos
    groups = []
    for grp, _ in columns:
        if groups and groups[-1][0] == grp:
            groups[-1][1] += 1
        else:
            groups.append([grp, 1])  # [nome, span]

    # Cabeçalho 1: grupos (primeira célula vazia)
    th_groups = ["<th class='stub'></th>"]
    for grp, span in groups:
        label = grp if grp else ""  # Total não mostra nome de grupo
        th_groups.append(f"<th class='grp' colspan='{span}'>{label}</th>")

    # Cabeçalho 2: rótulos das colunas
    th_cols = ["<th class='stub'></th>"]
    for _grp, col in columns:
        th_cols.append(f"<th class='col'>{col}</th>")

    # Linhas
    trs = []
    for rlbl, cells in rows:
        tds = [f"<td class='rowlbl'>{rlbl}</td>"] + [f"<td class='cell'>{c}</td>" for c in cells]
        trs.append("<tr>" + "".join(tds) + "</tr>")

    # Base
    base_tr = "<tr class='base'><td class='rowlbl'>Base</td>" + "".join(
        f"<td class='cell'>{b}</td>" for b in base_cells
    ) + "</tr>"

    return f"""
    <div class="tbl">
      <table class="ctab">
        <thead>
          <tr>{''.join(th_groups)}</tr>
          <tr>{''.join(th_cols)}</tr>
        </thead>
        <tbody>
          {''.join(trs)}
          {base_tr}
        </tbody>
      </table>
    </div>
    """


# ===========================================================
# Múltiplas respostas (ponderadas)
# ===========================================================
def cross_multi(df, multi_vars, by_var=None, selected_value="1",
                treat_nonzero=False, decimals=1, meta=None,
                weight=None, base_weighted=True):
    """
    Retorna lista [(title, html)] com uma tabela (Total + opcional BY).
    Cada variável do conjunto vira LINHA.
    """
    if not multi_vars:
        return [("Múltiplas respostas", "<div class='tbl'><i>Selecione variáveis do conjunto múltiplo.</i></div>")]

    w_all = _weight_series(df, weight)
    opt_labels = [_var_label(meta, v) for v in multi_vars]

    def _marked(series):
        if treat_nonzero:
            return series.notna() & (series != 0)
        try:
            return (series.astype(str) == str(selected_value))
        except Exception:
            return (series == selected_value)

    # Monta estrutura WIDE (Total + categorias do BY em colunas)
    columns = [("", "Total")]
    col_masks = [pd.Series(True, index=df.index)]
    if by_var:
        vmap = _value_labels(meta, by_var) or {}
        if vmap:
            by_codes = list(vmap.keys())
            by_labels = [vmap[c] for c in by_codes]
        else:
            by_codes = list(pd.unique(df[by_var].dropna()))
            by_labels = [str(c) for c in by_codes]
        for code, lab in zip(by_codes, by_labels):
            columns.append((_var_label(meta, by_var), lab))
            col_masks.append(df[by_var] == code)

    # Denominadores por coluna
    denoms = [w_all[m].sum() for m in col_masks]

    rows = []
    for v, rlbl in zip(multi_vars, opt_labels):
        row_cells = []
        sel = _marked(df[v])
        for j, mcol in enumerate(col_masks):
            num = w_all[sel & mcol].sum()
            denom = denoms[j]
            pct = (num / denom * 100.0) if denom and denom > 0 else np.nan
            row_cells.append(_fmt_pct(pct, decimals))
        rows.append((rlbl, row_cells))

    # Base
    base_cells = []
    for mcol in col_masks:
        base_val = int(round(w_all[mcol].sum())) if base_weighted else int(mcol.sum())
        base_cells.append(str(base_val))

    tdict = {
        "title": _var_label(meta, multi_vars[0]) if multi_vars else "Múltiplas",
        "caption": f"Pergunta: {_var_label(meta, multi_vars[0]) if multi_vars else ''}",
        "columns": [c[1] for c in columns],
        "rows": rows,
        "base": base_cells,
        "_columns_grouped": columns
    }
    return [(tdict["title"], render_table_html_wide(tdict))]


# =========================
# Exportar para Excel (WIDE) — ATUALIZADO
# =========================
def build_excel_bytes(df, meta, params):
    """
    Exporta em formato WIDE (Total + todos os BYs) com dois cabeçalhos.
    - Percentuais como NÚMERO (ex.: 9.1), não como texto
    - Base como inteiro (ex.: 2000)
    - Cabeçalho com quebra automática e largura auto-ajustada
    - Rótulos (1a coluna) alinhados à esquerda
    - Borda espessa apenas no contorno externo; internas finas
    """
    import xlsxwriter
    import math
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        wb = xw.book

        def num_fmt(decimals: int):
            return "0" if decimals <= 0 else ("0." + ("0" * decimals))

        def write_wide_sheet(name, table_dict, decimals=1):
            ws = wb.add_worksheet(name[:31])

            cols_grouped = table_dict.get("_columns_grouped") or [("", c) for c in table_dict["columns"]]
            rows = table_dict["rows"]
            base_cells = table_dict["base"]

            # ====== dimensões ======
            ncols = len(cols_grouped)
            ndata = len(rows)
            r0 = 2
            r_base = r0 + ndata
            last_row = r_base
            last_col = ncols

            # ====== construir grupos para o cabeçalho ======
            groups = []
            for grp, _ in cols_grouped:
                if groups and groups[-1][0] == grp:
                    groups[-1][1] += 1
                else:
                    groups.append([grp, 1])

            # ====== Formatação e escrita da grade principal (sem o cabeçalho de grupo) ======
            # O laço começa em r=1 para pular a linha de grupos, que será tratada depois
            for r in range(1, last_row + 1):
                for c in range(0, last_col + 1):
                    val = None
                    # Lógica para obter o valor da célula (val)
                    if r == 1: # Linha de títulos de coluna
                        val = "" if c == 0 else cols_grouped[c - 1][1]
                    elif r == r_base: # Linha da Base
                        if c == 0:
                            val = "Base"
                        else:
                            try:
                                val = int(float(str(base_cells[c - 1]).replace(",", ".")))
                            except (ValueError, IndexError):
                                val = None
                    else: # Linhas de dados (percentuais)
                        if c == 0:
                            val = rows[r - r0][0]
                        else:
                            cell_val = rows[r - r0][1][c - 1]
                            if isinstance(cell_val, str) and cell_val.strip():
                                val = float(cell_val.replace(",", "."))
                            else:
                                val = cell_val if pd.notna(cell_val) else None
                    
                    # Lógica de bordas
                    left = 2 if c == 0 else 1
                    right = 2 if c == last_col else 1
                    top = 2 if r == 1 else 1 # Borda superior espessa na primeira linha visível (títulos)
                    bottom = 2 if r == last_row else 1

                    # Monta o dicionário de formatação
                    kwargs = {
                        "bold": (r == 1 or r == r_base),
                        "align": "center", "valign": "vcenter",
                        "text_wrap": (r == 1),
                        "left": left, "right": right, "top": top, "bottom": bottom
                    }
                    if r == r_base and c > 0:
                        kwargs["num_format"] = "0"
                    elif c > 0 and r > 1:
                        kwargs["num_format"] = num_fmt(decimals)
                    if c == 0 and r > 0:
                        kwargs["align"] = "left"
                        kwargs.pop("num_format", None)

                    # Escreve a célula com o valor e formato
                    ws.write(r, c, val, wb.add_format(kwargs))

            # ====== Escrita do cabeçalho de GRUPOS (linha 0) com células mescladas ======
            # Canto superior esquerdo (célula 0,0)
            corner_fmt = wb.add_format({'bold': True, 'top': 2, 'left': 2, 'bottom': 1, 'right': 1})
            ws.write(0, 0, "", corner_fmt)
            
            current_col = 1
            for i, (grp, span) in enumerate(groups):
                label = grp if grp else ""
                is_last_group = (i == len(groups) - 1)
                
                group_fmt = wb.add_format({
                    'bold': True, 'text_wrap': True, 'align': 'center', 'valign': 'vcenter',
                    'top': 2,
                    'bottom': 1,
                    'left': 1,
                    'right': 2 if is_last_group else 1,
                })
                
                if span > 1:
                    ws.merge_range(0, current_col, 0, current_col + span - 1, label, group_fmt)
                else:
                    ws.write(0, current_col, label, group_fmt)
                current_col += span

            # ====== Ajuste de Largura e Altura de Colunas/Linhas ======
            max_lbl_len = max([len(str(x[0])) for x in rows] + [len("Base"), 8])
            ws.set_column(0, 0, min(max(18, int(max_lbl_len * 1.05)), 48))
            for j, (_g, ctitle) in enumerate(cols_grouped, start=1):
                est_val_len = max(len(ctitle), 5 + (1 if decimals > 0 else 0) + decimals)
                ws.set_column(j, j, min(max(8, int(est_val_len * 1.05)), 18))
            
            ws.set_row(0, 28)
            ws.set_row(1, 28)

        # ==== Reconstrói tabelas conforme os parâmetros usados na tela ====
        if params.get("mode", "simple") == "simple":
            targets = params.get("targets") or []
            by_vars = params.get("by_vars") or []
            weight = params.get("weight_var") or ("peso" if "peso" in df.columns else "")
            percent_axis = params.get("percent_axis", "col")
            base_weighted = bool(params.get("base_weighted", True))
            decimals = int(params.get("decimals", 1) or 1)
            filt = (params.get("filter_expr") or "").strip()

            sub = df
            if filt:
                try:
                    sub = df.query(filt, engine="python")
                except Exception:
                    pass

            for t in targets:
                t_wide = compute_table_wide(sub, t, by_vars=by_vars, weight=weight,
                                            percent_axis=percent_axis, base_weighted=base_weighted,
                                            decimals=decimals, meta=meta)
                write_wide_sheet(_var_label(meta, t)[:31], t_wide, decimals=decimals)

        else:
            # múltiplas
            multi_vars = params.get("multi_vars") or []
            by_var = params.get("by_var") or None
            if by_var == "":
                by_var = None
            selected_value = params.get("selected_value", "1")
            treat_nonzero = bool(params.get("treat_nonzero", False))
            weight = params.get("weight_var") or ("peso" if "peso" in df.columns else "")
            base_weighted = bool(params.get("base_weighted", True))
            decimals = int(params.get("decimals", 1) or 1)
            filt = (params.get("filter_expr") or "").strip()

            sub = df
            if filt:
                try:
                    sub = df.query(filt, engine="python")
                except Exception:
                    pass

            tables = cross_multi(sub, multi_vars, by_var=by_var, selected_value=selected_value,
                                 treat_nonzero=treat_nonzero, decimals=decimals, meta=meta,
                                 weight=weight, base_weighted=base_weighted)
            for title, _html in tables:
                # reconstrução numérica semelhante ao HTML wide
                columns = [("", "Total")]
                if by_var:
                    vmap = _value_labels(meta, by_var) or {}
                    if vmap:
                        by_codes = list(vmap.keys())
                        by_labels = [vmap[c] for c in by_codes]
                    else:
                        by_codes = list(pd.unique(sub[by_var].dropna()))
                        by_labels = [str(c) for c in by_codes]
                    for lab in by_labels:
                        columns.append((_var_label(meta, by_var), lab))

                w_all = _weight_series(sub, weight)
                col_masks = [pd.Series(True, index=sub.index)]
                if by_var:
                    for code in (by_codes if vmap else by_codes):
                        col_masks.append(sub[by_var] == code)
                denoms = [w_all[m].sum() for m in col_masks]

                rows = []
                for v in multi_vars:
                    lab = _var_label(meta, v)
                    s = sub[v]
                    if treat_nonzero:
                        marked = s.notna() & (s != 0)
                    else:
                        try:
                            marked = (s.astype(str) == str(selected_value))
                        except Exception:
                            marked = (s == selected_value)
                    row_cells = []
                    for j, mcol in enumerate(col_masks):
                        num = w_all[marked & mcol].sum()
                        denom = denoms[j]
                        pct = (num / denom * 100.0) if denom and denom > 0 else np.nan
                        row_cells.append("" if pd.isna(pct) else f"{pct:.{decimals}f}")
                    rows.append((lab, row_cells))

                base_cells = [str(int(round(w_all[m].sum()))) for m in col_masks]
                tdict = {
                    "title": title,
                    "caption": f"Pergunta: {title}",
                    "columns": [c[1] for c in columns],
                    "rows": rows,
                    "base": base_cells,
                    "_columns_grouped": columns
                }
                write_wide_sheet(title[:31], tdict, decimals=decimals)

    buf.seek(0)
    return buf



# =========================
# Rotas
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("datafile")
        if not file or file.filename == "":
            flash("Envie um arquivo .sav/.zsav ou .csv")
            return redirect(url_for("index"))
        try:
            df, meta, name = read_dataset(file)
        except Exception as e:
            flash(f"Erro ao ler arquivo: {e}")
            return redirect(url_for("index"))
        token = str(uuid4())
        DATASETS[token] = {"df": df, "meta": meta, "name": name, "last_params": {}, "last_tables": []}
        return redirect(url_for("analyze", token=token))
    return render_template("index.html")


@app.route("/analyze/<token>", methods=["GET", "POST"])
def analyze(token):
    bundle = DATASETS.get(token)
    if not bundle:
        flash("Sessão expirada. Envie o arquivo novamente.")
        return redirect(url_for("index"))

    df, meta, name = bundle["df"], bundle["meta"], bundle["name"]
    vars_list = list(df.columns)

    tables_html = []
    params = bundle.get("last_params", {})
    last_tables = []

    try:
        if request.method == "POST":
            mode = request.form.get("mode", "simple")  # simple | multiple
            decimals = int(request.form.get("decimals", 1) or 1)

            # Peso: default "peso" se existir
            weight_var = request.form.get("weight_var") or ""
            if not weight_var and "peso" in df.columns:
                weight_var = "peso"

            percent_axis = request.form.get("percent_axis", "col")
            base_weighted = request.form.get("base_weighted") in ("on", "true", "1", True)

            # BYs e Targets
            by_vars = request.form.getlist("by_vars")  # lista
            targets = request.form.getlist("targets") if mode == "simple" else []

            # Filtro opcional
            filter_expr = (request.form.get("filter_expr") or "").strip()
            sub = df
            if filter_expr:
                try:
                    sub = df.query(filter_expr, engine="python")
                except Exception as e:
                    flash(f"Expressão de filtro inválida: {e}")
                    return render_template("analyze.html", token=token, name=name, vars_list=vars_list,
                                           tables=[], params=params, meta_available=meta is not None)

            if mode == "simple":
                if not targets:
                    flash("Selecione ao menos uma variável-alvo.")
                    return render_template("analyze.html", token=token, name=name, vars_list=vars_list,
                                           tables=tables_html, params=params, meta_available=meta is not None)

                # UMA TABELA ÚNICA POR ALVO (Total + TODOS OS BYs)
                for target in targets:
                    tdict = compute_table_wide(
                        sub, target,
                        by_vars=by_vars,
                        weight=weight_var,
                        percent_axis=percent_axis,
                        base_weighted=base_weighted,
                        decimals=decimals,
                        meta=meta
                    )
                    tables_html.append(render_table_html_wide(tdict))
                    last_tables.append(tdict)

                params = {"mode": mode, "targets": targets, "by_vars": by_vars, "decimals": decimals,
                          "weight_var": weight_var, "percent_axis": percent_axis, "base_weighted": base_weighted,
                          "filter_expr": filter_expr}

            else:
                # multiple (conjunto múltiplo)
                multi_vars = request.form.getlist("multi_vars")
                by_var = request.form.get("by_var") or None
                selected_value = request.form.get("selected_value", "1")
                treat_nonzero = request.form.get("treat_nonzero") in ("on", "true", "1", True)

                tables = cross_multi(sub, multi_vars, by_var=by_var, selected_value=selected_value,
                                     treat_nonzero=treat_nonzero, decimals=decimals, meta=meta,
                                     weight=weight_var, base_weighted=base_weighted)
                for _title, html in tables:
                    tables_html.append(html)

                params = {"mode": mode, "multi_vars": multi_vars, "by_var": by_var,
                          "selected_value": selected_value, "treat_nonzero": treat_nonzero,
                          "decimals": decimals, "weight_var": weight_var,
                          "base_weighted": base_weighted, "filter_expr": filter_expr}

            bundle["last_params"] = params
            bundle["last_tables"] = last_tables

    except Exception as e:
        traceback.print_exc()
        flash(f"Erro ao gerar tabelas: {e}")
        return render_template("analyze.html", token=token, name=name, vars_list=vars_list,
                               tables=[], params=params, meta_available=meta is not None)

    return render_template("analyze.html", token=token, name=name, vars_list=vars_list,
                           tables=tables_html, params=params, meta_available=meta is not None)


# ===== NOVA ROTA DE EXPORTAÇÃO (GET) =====
@app.route("/export_excel/<token>", methods=["GET"])
def export_excel(token):
    bundle = DATASETS.get(token)
    if not bundle:
        flash("Sessão expirada. Envie o arquivo novamente.")
        return redirect(url_for("index"))

    df, meta = bundle["df"], bundle["meta"]
    params = bundle.get("last_params", {})
    if not params:
        flash("Gere as tabelas antes de exportar.")
        return redirect(url_for("analyze", token=token))

    xlsx = build_excel_bytes(df, meta, params)
    return send_file(
        xlsx,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="tabelas.xlsx"
    )


if __name__ == "__main__":
    # Rode com: python app.py
    app.run(debug=True, host="0.0.0.0", port=5000)