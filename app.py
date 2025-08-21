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

# Limite seguro para largura (Total + categorias dos BYs) por “parte”
MAX_COLS_PER_PAGE = 300


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
        # pyreadstat exige path
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
# Helpers
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
    """
    if meta is None or var is None:
        return None

    if not isinstance(var, (str, int)):
        try:
            var = var.get("value") or var.get("name") or str(var)
        except Exception:
            var = str(var)

    vvl = getattr(meta, "value_labels", None) or {}

    # 1) rótulo diretamente por variável
    direct = vvl.get(var)
    if isinstance(direct, dict):
        return direct

    # 2/3) via variable_value_labels
    var_to_labelset = getattr(meta, "variable_value_labels", None) or {}
    labelset = var_to_labelset.get(var)

    if isinstance(labelset, dict):
        return labelset
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
    return f"{x:.{dec}f}".replace(".", ",")


def _by_var_colcount(df, by, meta):
    vmap = _value_labels(meta, by) or {}
    return len(vmap) if vmap else int(df[by].nunique(dropna=True))


def _split_by_vars_into_pages(by_vars, df, meta, max_cols):
    """
    Divide BYs em “partes” para que (Total + soma das categorias) não ultrapasse max_cols.
    """
    pages, cur, curcols = [], [], 1  # +1 para Total
    for by in by_vars or []:
        cnt = max(1, _by_var_colcount(df, by, meta))
        if cur and curcols + cnt > max_cols:
            pages.append(cur)
            cur, curcols = [], 1
        cur.append(by)
        curcols += cnt
    if cur or not pages:
        pages.append(cur)  # se não houver BYs => [[]]
    return pages


# =========================
# Tabela LARGA (Total + todos BYs em colunas)
# =========================
def compute_table_wide(
    df, target, by_vars=None, weight=None,
    percent_axis="col", base_weighted=True, decimals=1, meta=None
):
    # normalização
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

    # linhas (target)
    targ_vmap = _value_labels(meta, target) or {}
    if targ_vmap:
        targ_codes = list(targ_vmap.keys())
        targ_row_labels = [targ_vmap[c] for c in targ_codes]
    else:
        targ_codes = list(pd.unique(df[target].dropna()))
        targ_row_labels = [str(c) for c in targ_codes]

    # colunas
    columns = [("", "Total")]
    col_masks = [pd.Series(True, index=df.index)]
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

    denoms = [(w_all[m].sum() if percent_axis == "col" else None) for m in col_masks]

    rows = []
    for r_code, r_label in zip(targ_codes, targ_row_labels):
        row_cells = []
        row_mask = (df[target] == r_code)
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

    base_cells = []
    for mcol in col_masks:
        base_val = int(round(w_all[mcol].sum())) if base_weighted else int(mcol.sum())
        base_cells.append(str(base_val))

    return {
        "title": _var_label(meta, target),
        "caption": f"Pergunta: {_var_label(meta, target)}",
        "columns": [c[1] for c in columns],
        "rows": rows,
        "base": base_cells,
        "_columns_grouped": columns
    }


def render_table_html_wide(table_dict):
    """
    Renderer HTML (2 linhas de header: grupos e rótulos).
    """
    columns = table_dict.get("_columns_grouped") or [("", c) for c in table_dict["columns"]]
    rows = table_dict["rows"]
    base_cells = table_dict["base"]

    # grupos
    groups = []
    for grp, _ in columns:
        if groups and groups[-1][0] == grp:
            groups[-1][1] += 1
        else:
            groups.append([grp, 1])

    # header 1
    th_groups = ["<th class='stub'></th>"]
    for grp, span in groups:
        label = grp if grp else "Total"
        th_groups.append(f"<th class='grp' colspan='{span}'>{label}</th>")

    # header 2
    th_cols = ["<th class='stub'></th>"]
    for _grp, col in columns:
        th_cols.append(f"<th class='col'>{col}</th>")

    # linhas
    trs = []
    for rlbl, cells in rows:
        tds = [f"<td class='rowlbl'>{rlbl}</td>"] + [f"<td class='cell'>{c}</td>" for c in cells]
        trs.append("<tr>" + "".join(tds) + "</tr>")

    # base
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


# =========================
# Múltiplas respostas
# =========================
def cross_multi(df, multi_vars, by_var=None, selected_value="1",
                treat_nonzero=False, decimals=1, meta=None,
                weight=None, base_weighted=True):
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
# Exportar Excel — UMA planilha, todas as tabelas empilhadas
# =========================
def build_excel_bytes(df, meta, params):
    import xlsxwriter, re
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        wb = xw.book

        # --- cria uma única planilha ---
        def _sanitize(name: str) -> str:
            name = (name or "Planilha")
            name = re.sub(r"[:\\/?*\[\]\\]", " ", str(name)).strip()
            return (name or "Planilha")[:31]
        ws = wb.add_worksheet(_sanitize("Cruzamentos"))

        # formatos (cache)
        fmt_cache = {}
        def get_fmt(**kwargs):
            key = tuple(sorted(kwargs.items()))
            fmt = fmt_cache.get(key)
            if fmt is None:
                fmt = wb.add_format(kwargs)
                fmt_cache[key] = fmt
            return fmt

        def num_fmt(decimals: int):
            return "0" if decimals <= 0 else ("0." + ("0"*decimals))

        def _title_for_table(table_dict):
            axis_txt = {"col": "% por coluna", "row": "% por linha", "all": "% do total"} \
                       .get(params.get("percent_axis", "col"), "%")
            peso_txt = f"Peso: {params.get('weight_var')}" if params.get("weight_var") else "Sem peso"
            filtro = (params.get("filter_expr") or "").strip()
            filtro_txt = f" | Filtro: {filtro}" if filtro else ""

            cols_grouped = table_dict.get("_columns_grouped") or [("", c) for c in table_dict["columns"]]
            by_names = []
            for grp, _ in cols_grouped:
                g = grp if grp else "Total"
                if g != "Total" and g not in by_names:
                    by_names.append(g)
            by_txt = f"BY: {', '.join(by_names)}" if by_names else "BY: Total"

            return f"{table_dict.get('title') or 'Tabela'}  —  {by_txt}  •  {axis_txt}  •  {peso_txt}{filtro_txt}"

        # escreve uma tabela a partir de start_row; retorna a próxima linha livre (com 2 linhas em branco)
        def write_wide_table(ws, start_row, table_dict, decimals=1):
            title_text = _title_for_table(table_dict)
            cols_grouped = table_dict.get("_columns_grouped") or [("", c) for c in table_dict["columns"]]
            rows = table_dict["rows"]
            base_cells = table_dict["base"]

            ncols = len(cols_grouped)
            ndata = len(rows)

            r_title  = start_row
            r_groups = start_row + 1
            r_cols   = start_row + 2
            r0       = start_row + 3
            r_base   = r0 + ndata
            last_row = r_base
            last_col = ncols

            # Título
            ws.merge_range(r_title, 0, r_title, last_col, title_text,
                           get_fmt(bold=True, font_size=14, align='center', valign='vcenter'))

            # Grupos
            groups = []
            for grp, _ in cols_grouped:
                g = grp if grp else "Total"
                if groups and groups[-1][0] == g:
                    groups[-1][1] += 1
                else:
                    groups.append([g, 1])

            ws.write(r_groups, 0, "", get_fmt(bold=True, top=2, left=2, bottom=1, right=1))
            current_col = 1
            for i, (grp, span) in enumerate(groups):
                is_last = (i == len(groups) - 1)
                gf = get_fmt(bold=True, text_wrap=True, align='center', valign='vcenter',
                             top=2, bottom=1, left=1, right=(2 if is_last else 1))
                if span > 1:
                    ws.merge_range(r_groups, current_col, r_groups, current_col + span - 1, grp, gf)
                else:
                    ws.write(r_groups, current_col, grp, gf)
                current_col += span

            # Cabeçalhos + dados
            for r in range(r_cols, last_row + 1):
                for c in range(0, last_col + 1):
                    val = None
                    if r == r_cols:
                        val = "" if c == 0 else cols_grouped[c - 1][1]
                    elif r == r_base:
                        if c == 0:
                            val = "Base"
                        else:
                            try:
                                val = int(float(str(base_cells[c - 1]).replace(",", ".")))
                            except (ValueError, IndexError):
                                val = None
                    else:
                        if c == 0:
                            val = rows[r - r0][0]
                        else:
                            cell_val = rows[r - r0][1][c - 1]
                            if isinstance(cell_val, str) and cell_val.strip():
                                try:
                                    val = float(cell_val.replace(",", "."))
                                except Exception:
                                    val = None
                            else:
                                val = cell_val if pd.notna(cell_val) else None

                    left = 2 if c == 0 else 1
                    right = 2 if c == last_col else 1
                    top = 2 if r == r_cols else 1
                    bottom = 2 if r == last_row else 1

                    kwargs = {
                        "bold": (r == r_cols or r == r_base),
                        "align": "center", "valign": "vcenter",
                        "text_wrap": (r == r_cols),
                        "left": left, "right": right, "top": top, "bottom": bottom
                    }
                    if r == r_base and c > 0:
                        kwargs["num_format"] = "0"
                    elif c > 0 and r > r_cols:
                        kwargs["num_format"] = num_fmt(decimals)
                    if c == 0 and r >= r_cols:
                        kwargs["align"] = "left"
                        kwargs.pop("num_format", None)

                    ws.write(r, c, val, get_fmt(**kwargs))

            # larguras
            max_lbl_len = max([len(str(x[0])) for x in rows] + [len("Base"), 8]) if rows else 8
            ws.set_column(0, 0, min(max(18, int(max_lbl_len * 1.05)), 48))
            for j, (_g, ctitle) in enumerate(cols_grouped, start=1):
                est = max(len(ctitle), 5 + (1 if decimals > 0 else 0) + decimals)
                ws.set_column(j, j, min(max(8, int(est * 1.05)), 18))

            # congela painéis na primeira tabela
            if start_row == 0:
                ws.freeze_panes(r0, 1)

            return last_row + 3  # 2 linhas em branco entre as tabelas

        # Empilhar todas as tabelas na MESMA planilha
        next_row = 0

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

            pages = _split_by_vars_into_pages(by_vars, sub, meta, MAX_COLS_PER_PAGE)

            for t in targets:
                for page in pages:
                    t_wide = compute_table_wide(
                        sub, t,
                        by_vars=page,
                        weight=weight,
                        percent_axis=percent_axis,
                        base_weighted=base_weighted,
                        decimals=decimals,
                        meta=meta
                    )
                    next_row = write_wide_table(ws, next_row, t_wide, decimals=decimals)

        else:
            # múltiplas – também na mesma planilha
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

            columns = [("", "Total")]
            w_all = _weight_series(sub, weight)
            col_masks = [pd.Series(True, index=sub.index)]
            if by_var:
                vmap = _value_labels(meta, by_var) or {}
                if vmap:
                    by_codes = list(vmap.keys())
                    by_labels = [vmap[c] for c in by_codes]
                else:
                    by_codes = list(pd.unique(sub[by_var].dropna()))
                    by_labels = [str(c) for c in by_codes]
                for code, lab in zip(by_codes, by_labels):
                    columns.append((_var_label(meta, by_var), lab))
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
            tdict = {"title": "Múltiplas", "caption": "Múltiplas",
                     "columns": [c[1] for c in columns], "rows": rows,
                     "base": base_cells, "_columns_grouped": columns}
            next_row = write_wide_table(ws, next_row, tdict, decimals=decimals)

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
    n_rows = len(df)

    tables_html = []
    params = bundle.get("last_params", {})
    last_tables = []

    try:
        if request.method == "POST":
            mode = request.form.get("mode", "simple")  # simple | multiple
            decimals = int(request.form.get("decimals", 1) or 1)

            weight_var = request.form.get("weight_var") or ""
            if not weight_var and "peso" in df.columns:
                weight_var = "peso"

            percent_axis = request.form.get("percent_axis", "col")
            base_weighted = request.form.get("base_weighted") in ("on", "true", "1", True)

            # BYs e Targets
            by_vars = request.form.getlist("by_vars")
            targets = request.form.getlist("targets") if mode == "simple" else []

            # Filtro
            filter_expr = (request.form.get("filter_expr") or "").strip()
            sub = df
            if filter_expr:
                try:
                    sub = df.query(filter_expr, engine="python")
                except Exception as e:
                    flash(f"Expressão de filtro inválida: {e}")
                    return render_template("analyze.html", token=token, name=name, vars_list=vars_list,
                                           n_rows=n_rows, tables=[], params=params, meta_available=meta is not None)

            if mode == "simple":
                if not targets:
                    flash("Selecione ao menos uma variável-alvo.")
                    return render_template("analyze.html", token=token, name=name, vars_list=vars_list,
                                           n_rows=n_rows, tables=tables_html, params=params, meta_available=meta is not None)

                # dividir BYs em partes (para não estourar largura)
                pages = _split_by_vars_into_pages(by_vars, sub, meta, MAX_COLS_PER_PAGE)
                for target in targets:
                    for idx, page in enumerate(pages, start=1):
                        tdict = compute_table_wide(
                            sub, target,
                            by_vars=page,
                            weight=weight_var,
                            percent_axis=percent_axis,
                            base_weighted=base_weighted,
                            decimals=decimals,
                            meta=meta
                        )
                        heading = f"<h3 class='part'>{_var_label(meta, target)} — parte {idx}</h3>" if len(pages) > 1 else ""
                        tables_html.append(heading + render_table_html_wide(tdict))
                        last_tables.append(tdict)

                params = {"mode": mode, "targets": targets, "by_vars": by_vars, "decimals": decimals,
                          "weight_var": weight_var, "percent_axis": percent_axis, "base_weighted": base_weighted,
                          "filter_expr": filter_expr}

            else:
                # múltiplas
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
                               n_rows=n_rows, tables=[], params=params, meta_available=meta is not None)

    return render_template("analyze.html", token=token, name=name, vars_list=vars_list,
                           n_rows=n_rows, tables=tables_html, params=params, meta_available=meta is not None)


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

    try:
        xlsx = build_excel_bytes(df, meta, params)
    except Exception as e:
        traceback.print_exc()
        flash(f"Erro na exportação: {e}")
        return redirect(url_for("analyze", token=token))

    return send_file(
        xlsx,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="tabelas.xlsx"
    )


@app.route("/reset/<token>")
def reset(token):
    DATASETS.pop(token, None)
    flash("Base descartada. Carregue outro arquivo.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Rode com: python app.py
    app.run(debug=True, host="0.0.0.0", port=5000)
