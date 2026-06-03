"""
SICOP Adjudicaciones Dashboard
Scrutinize the adjudicaciones process across institutions.
Reads from parquet files (no database required).
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Análisis de Adjudicaciones — SICOP",
    page_icon="🔍",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

# ── product category mapping ─────────────────────────────────────────────────

import re

CATEGORY_RULES = [
    ("Infraestructura y Construcción", [
        r"construcci[oó]n", r"acera", r"paviment", r"asfalt", r"muro\b", r"puente",
        r"alcantarill", r"obra[s]?\b", r"cimenta", r"demolici", r"remodelaci",
        r"reparaci[oó]n.*(edifici|inmueble|infraestructur)", r"techado", r"cubierta.*techo",
        r"nave industrial", r"dise[ñn]o.*plano", r"topograf",
        r"acondicionamiento.*sal[oó]n", r"mejora.*parque", r"gavion",
        r"instalaci[oó]n.*(red|el[eé]ctric|mec[aá]nic)", r"soldadura",
    ]),
    ("Tecnología e Informática", [
        r"comput", r"servidor", r"laptop", r"notebook", r"portatil|portátil",
        r"software", r"licencia.*software", r"sistema.*inform",
        r"switch", r"router", r"firewall", r"ups\b", r"rack\b",
        r"imac\b", r"macbook", r"ipad\b", r"monitor\b",
        r"impresora", r"plotter", r"disco.*duro", r"webcam", r"web\s*cam",
        r"microsoft|office\s*365|sql.server", r"nube|cloud",
        r"red.*datos|fibra.*[oó]ptica|cableado.*estructurado",
        r"cibersegur|antivirus|fortinet|forti\w+", r"backup|respaldo.*autom",
        r"docking\b", r"arcgis|autocad|cad\b|zwcad", r"adobe",
        r"milestone|alarm\s*center", r"blip\b", r"elisiam",
        r"hosting|hospedaje.*nube", r"dominio.*web",
        r"firma.*digital", r"punto.*acceso.*inal[aá]mbric",
        r"tel[eé]fono.*ip|tel[eé]fono.*celular|iphone",
    ]),
    ("Vehículos y Transporte", [
        r"veh[ií]culo", r"cami[oó]n", r"automóvil|autom[oó]vil",
        r"motocicleta", r"flotilla", r"arrendamiento.*veh",
        r"gr[uú]a\b", r"remolque", r"llanta", r"neum[aá]tico",
        r"combustible|gasolina|di[eé]sel", r"lubricant",
        r"taller.*mec[aá]nic|mec[aá]nica.*automotriz",
        r"repuesto.*vehic", r"alquiler.*cami[oó]n",
    ]),
    ("Seguridad y Vigilancia", [
        r"seguridad", r"vigilancia", r"alarma", r"c[aá]mara.*seguridad",
        r"monitoreo|cctv", r"arma.*fuego|munici[oó]n",
        r"polic[ií]a", r"chaleco.*antibal", r"detector",
        r"control.*acceso", r"barrera.*autom",
    ]),
    ("Servicios Profesionales", [
        r"consultor[ií]a", r"asesor[ií]a", r"servicios.*profesionales",
        r"abogad[oa]|legal|jur[ií]dic", r"auditor[ií]a",
        r"contab|nicsp", r"notari", r"peritaje",
        r"estudio.*t[eé]cnico|estudio.*factibilidad",
        r"dise[ñn]o.*arquitect", r"arquitect",
        r"ingenier[ií]a.*civil|ingenier[ií]a.*el[eé]ctric",
        r"plan.*regulador|plan.*desarrollo",
    ]),
    ("Mantenimiento y Reparación", [
        r"mantenimiento", r"reparaci[oó]n", r"correctivo|preventivo",
        r"limpieza", r"fumigaci|plaga|desinfec",
        r"pintura.*edifici|pintura.*inmueble", r"fontaner",
        r"aire.*acondicionado.*manten", r"ascensor",
        r"jardin|chapea|poda|[aá]reas.*verdes",
    ]),
    ("Materiales y Suministros", [
        r"papel\b|papel.*bond", r"tinta\b|t[oó]ner", r"cartucho",
        r"material.*oficina", r"suministro", r"tuber[ií]a",
        r"pintura\b", r"cemento|concreto|bloques",
        r"herramienta", r"ferret", r"tornillo|clavo|tuerca",
        r"uniforme|indumentaria|vestuario|camisa",
        r"recipiente|basurer|contenedor",
        r"se[ñn]al[ée]tica|r[oó]tulo", r"letras.*volum[eé]tric",
    ]),
    ("Salud y Bienestar Social", [
        r"salud", r"m[eé]dic[oa]", r"hospital", r"cl[ií]nica",
        r"psicol[oó]g", r"rehabilitaci", r"discapacidad",
        r"esterilizaci[oó]n|castraci[oó]n", r"vacun",
        r"adulto.*mayor", r"ni[ñn]ez|cecudi",
        r"social\b|bienestar|vulnerab",
        r"mascarilla|equipo.*protecci[oó]n.*personal",
    ]),
    ("Capacitación y Eventos", [
        r"capacitaci[oó]n", r"taller\b", r"curso\b", r"congreso",
        r"seminario", r"charla\b", r"formaci[oó]n",
        r"evento|festival|feria\b|foro\b",
        r"alquiler.*hotel|alquiler.*instalacion|alquiler.*sal[oó]n",
        r"team.building|log[ií]stica.*evento",
        r"patrocin", r"rendici[oó]n.*cuentas",
    ]),
    ("Comunicación y Publicidad", [
        r"publicidad|public.*medios", r"campa[ñn]a.*divulgaci",
        r"medios.*comunicaci[oó]n", r"televisi[oó]n|tv\b|canal\b",
        r"radio\b|prensa\b|peri[oó]dico",
        r"dise[ñn]o.*gr[aá]fico|impresi[oó]n.*material",
        r"redes.*sociales", r"comunicaci[oó]n.*institucional",
        r"correo.*masivo",
    ]),
    ("Alquiler de Maquinaria y Equipos", [
        r"alquiler.*maquinaria", r"alquiler.*equipo",
        r"retroexcavadora|excavadora|compactador",
        r"hidrovaciador|tragante|pozo",
        r"arrendamiento.*impresora|arrendamiento.*comput",
        r"arrendamiento.*radio",
        r"caba[ñn]a.*sanitaria",
    ]),
    ("Alimentación", [
        r"aliment", r"comida\b", r"bebida", r"almuerzo",
        r"desayuno", r"catering|servicio.*comedor",
        r"jugo|quequit|refrescos", r"percolador|caf[eé]\b",
    ]),
    ("Medio Ambiente y Residuos", [
        r"residuo|desecho|reciclaje|reciclado",
        r"ambiente|ambiental|ecol[oó]g",
        r"reforestaci|pl[aá]ntula|vivero|[aá]rbol",
        r"agua.*potable|acueducto",
        r"contaminaci|emisiones",
        r"manejo.*residuo",
    ]),
]

def categorize_description(desc: str) -> str:
    if not desc or not isinstance(desc, str):
        return "Sin categoría"
    text = desc.lower().replace("\xa0", " ")
    for category, patterns in CATEGORY_RULES:
        for pat in patterns:
            if re.search(pat, text):
                return category
    return "Otros"

INSTITUTIONS = {
    "Municipalidad de San José (3014042058)": "3014042058",
    "Municipalidad de Montes de Oca (3014042053)": "3014042053",
}

# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_tables(cedula: str) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ["adjudicaciones", "contratos", "ordenes_pedido",
                  "detalle_pliego", "proveedores"]:
        path = DATA_DIR / f"{cedula}_{name}.parquet"
        tables[name] = pd.read_parquet(path)
    return tables


def prepare_data(tables: dict[str, pd.DataFrame]):
    adj = tables["adjudicaciones"].copy()
    con = tables["contratos"].copy()
    op  = tables["ordenes_pedido"].copy()
    dp  = tables["detalle_pliego"].copy()
    prov = tables["proveedores"].copy()

    # ── amounts ──
    op["monto"]     = pd.to_numeric(op["monto orden pedido"], errors="coerce").fillna(0)
    op["cambio"]    = pd.to_numeric(op["tipo cambio crc"], errors="coerce").fillna(1)
    op["monto_crc"] = np.where(
        op["tipo moneda"].str.upper() == "CRC", op["monto"], op["monto"] * op["cambio"]
    )

    # ── dates ──
    adj["fecha_adj"] = pd.to_datetime(adj["fecha adj firme"], errors="coerce")
    adj["fecha_com"] = pd.to_datetime(adj.get("fecha comunicación", pd.Series(dtype="object")),
                                      errors="coerce")
    dp["descripción"] = dp["descripción"].fillna("").str.replace("\xa0", " ")

    # ── 1. amounts per procedure (from ordenes_pedido) ──
    op_proc = (
        op.groupby("número procedimiento")
        .agg(monto_total_crc=("monto_crc", "sum"),
             n_lineas_op=("monto_crc", "count"))
        .reset_index()
        .rename(columns={"número procedimiento": "_proc"})
    )

    # ── 2. proveedores per procedure (contratos → proveedores) ──
    con_prov = con.merge(prov[["cédula proveedor", "nombre proveedor",
                               "tipo proveedor", "tamaño proveedor", "provincia"]],
                         on="cédula proveedor", how="left")

    prov_proc = (
        con_prov.groupby("nro procedimiento")
        .agg(
            proveedores=("nombre proveedor",
                         lambda s: " | ".join(sorted(s.dropna().unique()))),
            cedulas_prov=("cédula proveedor",
                          lambda s: " | ".join(sorted(s.dropna().unique()))),
            n_contratos=("nro contrato", "nunique"),
        )
        .reset_index()
        .rename(columns={"nro procedimiento": "_proc"})
    )

    # ── 3. description + tipo from detalle_pliego ──
    dp["categoría"] = dp["descripción"].apply(categorize_description)
    dp_uniq = (
        dp.sort_values("número procedimiento")
        .drop_duplicates(subset="número procedimiento")
        [["número procedimiento", "descripción", "categoría", "tipo procedimiento",
          "modalidad procedimiento", "nombre unidad compra"]]
        .rename(columns={"número procedimiento": "_proc"})
    )

    # ── 4. enrich adjudicaciones ──
    enriched = (
        adj
        .merge(op_proc,   left_on="número de procedimiento", right_on="_proc", how="left")
        .drop(columns=["_proc"], errors="ignore")
        .merge(prov_proc, left_on="número de procedimiento", right_on="_proc", how="left")
        .drop(columns=["_proc"], errors="ignore")
        .merge(dp_uniq,   left_on="número de procedimiento", right_on="_proc", how="left")
        .drop(columns=["_proc"], errors="ignore")
    )
    enriched["monto_total_crc"] = enriched["monto_total_crc"].fillna(0)
    enriched["n_contratos"]     = enriched["n_contratos"].fillna(0).astype(int)

    # ── 5. proveedor‑level amounts (for top‑proveedor ranking) ──
    op_con = op.merge(
        con[["nro contrato", "cédula proveedor"]],
        left_on="no contrato", right_on="nro contrato", how="left",
    )
    op_con = op_con.merge(
        prov[["cédula proveedor", "nombre proveedor", "tipo proveedor",
              "tamaño proveedor", "provincia"]],
        on="cédula proveedor", how="left",
    )

    return enriched, op_con, op


# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🔍 Adjudicaciones SICOP")
st.sidebar.markdown("---")

sel_inst = st.sidebar.selectbox("Institución", list(INSTITUTIONS.keys()))
cedula = INSTITUTIONS[sel_inst]
inst_label = sel_inst.split(" (")[0]

tables = load_tables(cedula)
enriched, op_con, op_raw = prepare_data(tables)

# year filter
valid_dates = enriched["fecha_adj"].dropna()
if len(valid_dates):
    all_years = sorted(valid_dates.dt.year.unique())
    sel_years = st.sidebar.multiselect("Año(s)", all_years, default=all_years)
else:
    sel_years = []

# tipo procedimiento
tipos = sorted(enriched["tipo procedimiento"].dropna().unique())
sel_tipos = st.sidebar.multiselect("Tipo de Procedimiento", tipos, default=tipos)

# proveedor filter
all_proveedores = sorted(
    enriched["proveedores"].dropna()
    .str.split(r"\s*\|\s*").explode().str.strip()
    .loc[lambda s: s != ""].unique()
)
sel_proveedores = st.sidebar.multiselect("Proveedor(es)", all_proveedores,
                                          placeholder="Todos")

# categoría filter
categorias = sorted(enriched["categoría"].dropna().unique())
sel_categorias = st.sidebar.multiselect("Categoría de Producto", categorias,
                                         default=categorias)

# desierto
estado_opt = st.sidebar.radio("Estado", ["Todos", "Solo adjudicados", "Solo desiertos"])

# ── apply filters ──
df = enriched.copy()

if sel_years:
    df = df[df["fecha_adj"].dt.year.isin(sel_years) | df["fecha_adj"].isna()]

df = df[df["tipo procedimiento"].isin(sel_tipos) | df["tipo procedimiento"].isna()]

if sel_proveedores:
    mask = df["proveedores"].fillna("").apply(
        lambda p: any(s.strip() in p for s in sel_proveedores)
    )
    df = df[mask]

df = df[df["categoría"].isin(sel_categorias) | df["categoría"].isna()]

if estado_opt == "Solo adjudicados":
    df = df[df["desierto"].str.upper() == "N"]
elif estado_opt == "Solo desiertos":
    df = df[df["desierto"].str.upper() == "S"]

# filter op_con to match filtered procedures (all visualizations use this)
filtered_procs = set(df["número de procedimiento"].dropna().unique())
op_filtered = op_con[op_con["número procedimiento"].isin(filtered_procs)]

if sel_proveedores:
    op_filtered = op_filtered[op_filtered["nombre proveedor"].isin(sel_proveedores)]


# ── helpers ───────────────────────────────────────────────────────────────────

def fmt_crc(v):
    return f"₡{v:,.0f}" if pd.notna(v) and v != 0 else "—"


# ── header ────────────────────────────────────────────────────────────────────

st.title(f"📊 Adjudicaciones — {inst_label}")

# deduplicate by procedure for counts
proc_dedup = df.drop_duplicates(subset="número de procedimiento")
n_procs    = len(proc_dedup)
total_monto = proc_dedup["monto_total_crc"].sum()
with_monto  = (proc_dedup["monto_total_crc"] > 0).sum()
avg_monto   = proc_dedup.loc[proc_dedup["monto_total_crc"] > 0, "monto_total_crc"].mean() if with_monto else 0
n_prov      = (proc_dedup["proveedores"].dropna()
               .str.split(r"\s*\|\s*").explode().nunique())
n_desierto  = (proc_dedup["desierto"].str.strip().str.upper() == "S").sum()

st.caption(f"**{n_procs:,}** procedimientos · {len(df):,} registros de adjudicación")
st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
k1.metric("Procedimientos", f"{n_procs:,}")
k2.metric("Monto Total (₡)", fmt_crc(total_monto))
k3.metric("Proveedores", f"{n_prov:,}")
k4.metric("Desiertos", f"{n_desierto:,}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TOP PROVEEDORES BY AMOUNT
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("🏆 Top Proveedores por Monto Adjudicado")

prov_rank = (
    op_filtered
    .groupby(["cédula proveedor", "nombre proveedor", "tipo proveedor",
              "tamaño proveedor", "provincia"])
    .agg(monto_total=("monto_crc", "sum"), ordenes=("monto_crc", "count"))
    .reset_index()
    .sort_values("monto_total", ascending=False)
)
prov_rank = prov_rank[prov_rank["monto_total"] > 0]
prov_rank["nombre proveedor"] = prov_rank["nombre proveedor"].fillna(
    prov_rank["cédula proveedor"]
)

top_n = st.slider("Proveedores a mostrar", 10, 50, 25, key="prov_slider")
top_prov = prov_rank.head(top_n).copy()

col_chart, col_table = st.columns([3, 2])

with col_chart:
    fig_prov = px.bar(
        top_prov, y="nombre proveedor", x="monto_total", orientation="h",
        title=f"Top {top_n} Proveedores (₡)",
        labels={"monto_total": "Monto (₡)", "nombre proveedor": ""},
        color="monto_total",
        color_continuous_scale="Blues",
    )
    fig_prov.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(450, top_n * 28),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_prov, use_container_width=True)

with col_table:
    tbl_prov = top_prov[["nombre proveedor", "cédula proveedor", "monto_total",
                          "ordenes", "tipo proveedor", "tamaño proveedor",
                          "provincia"]].copy()
    tbl_prov.columns = ["Proveedor", "Cédula", "Monto Total (₡)", "Órdenes",
                         "Tipo", "Tamaño", "Provincia"]
    tbl_prov["Monto Total (₡)"] = tbl_prov["Monto Total (₡)"].apply(fmt_crc)
    tbl_prov.index = range(1, len(tbl_prov) + 1)
    st.dataframe(tbl_prov, width="stretch",
                 height=max(450, top_n * 28))

    csv_prov = prov_rank.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar lista completa proveedores",
                       csv_prov, "proveedores_por_monto.csv", "text/csv")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TOP 50 ADJUDICACIONES BY AMOUNT
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("💰 Top 50 Adjudicaciones por Monto")

proc_summary = proc_dedup.copy()

top50 = proc_summary[proc_summary["monto_total_crc"] > 0].head(50)

top50_disp = top50[[
    "número de procedimiento", "descripción", "categoría", "proveedores",
    "monto_total_crc", "tipo procedimiento", "modalidad procedimiento",
    "fecha_adj", "n_contratos",
]].copy()
top50_disp.columns = [
    "Procedimiento", "Descripción", "Categoría", "Proveedor(es)",
    "Monto (₡)", "Tipo", "Modalidad",
    "Fecha Adj. Firme", "Contratos",
]
top50_disp["Monto (₡)"]        = top50_disp["Monto (₡)"].apply(fmt_crc)
top50_disp["Fecha Adj. Firme"]  = pd.to_datetime(
    top50_disp["Fecha Adj. Firme"], errors="coerce"
).dt.strftime("%Y-%m-%d")
top50_disp.index = range(1, len(top50_disp) + 1)

st.dataframe(top50_disp, width="stretch", height=1200)

csv50 = top50_disp.to_csv(index=False).encode("utf-8")
st.download_button("📥 Descargar Top 50 (CSV)", csv50,
                   "top50_adjudicaciones.csv", "text/csv")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("📈 Análisis Temporal")

col_monthly, col_yearly = st.columns(2)

with col_monthly:
    dft = df[df["fecha_adj"].notna()].copy()
    dft["mes"] = dft["fecha_adj"].dt.to_period("M").astype(str)
    monthly = (
        dft.drop_duplicates("número de procedimiento")
        .groupby("mes")
        .agg(monto=("monto_total_crc", "sum"), n=("monto_total_crc", "count"))
        .reset_index()
    )
    fig_m = px.bar(monthly, x="mes", y="monto",
                   title="Monto Adjudicado por Mes",
                   labels={"mes": "", "monto": "Monto (₡)"})
    fig_m.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_m, use_container_width=True)

with col_yearly:
    dfy = df[df["fecha_adj"].notna()].copy()
    dfy["año"] = dfy["fecha_adj"].dt.year.astype(int)
    yearly = (
        dfy.drop_duplicates("número de procedimiento")
        .groupby("año")
        .agg(monto=("monto_total_crc", "sum"), n=("monto_total_crc", "count"))
        .reset_index()
    )
    fig_y = px.bar(yearly, x="año", y="n",
                   title="Cantidad de Procedimientos por Año",
                   labels={"año": "", "n": "Procedimientos"},
                   text="n")
    fig_y.update_traces(textposition="outside")
    st.plotly_chart(fig_y, use_container_width=True)

st.markdown("---")

# ── by type / modalidad ──
st.subheader("📋 Distribución por Tipo y Modalidad")
col_tipo, col_mod = st.columns(2)

with col_tipo:
    tipo_data = (
        proc_summary.groupby("tipo procedimiento")
        .agg(monto=("monto_total_crc", "sum"), n=("monto_total_crc", "count"))
        .reset_index()
        .sort_values("monto", ascending=False)
    )
    fig_t = px.pie(tipo_data, names="tipo procedimiento", values="monto",
                   title="Monto por Tipo de Procedimiento")
    st.plotly_chart(fig_t, use_container_width=True)

    tipo_tbl = tipo_data.copy()
    tipo_tbl.columns = ["Tipo Procedimiento", "Monto (₡)", "Cantidad"]
    tipo_tbl["Monto (₡)"] = tipo_tbl["Monto (₡)"].apply(fmt_crc)
    st.dataframe(tipo_tbl, width="stretch", hide_index=True)

with col_mod:
    mod_data = (
        proc_summary.groupby("modalidad procedimiento")
        .agg(monto=("monto_total_crc", "sum"), n=("monto_total_crc", "count"))
        .reset_index()
        .sort_values("monto", ascending=False)
    )
    fig_md = px.bar(mod_data, x="modalidad procedimiento", y="monto",
                    title="Monto por Modalidad",
                    labels={"modalidad procedimiento": "", "monto": "Monto (₡)"})
    fig_md.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_md, use_container_width=True)

    mod_tbl = mod_data.copy()
    mod_tbl.columns = ["Modalidad", "Monto (₡)", "Cantidad"]
    mod_tbl["Monto (₡)"] = mod_tbl["Monto (₡)"].apply(fmt_crc)
    st.dataframe(mod_tbl, width="stretch", hide_index=True)

st.markdown("---")

# ── by category ──
st.subheader("🏷️ Distribución por Categoría de Producto")
col_cat_chart, col_cat_table = st.columns([3, 2])

with col_cat_chart:
    cat_data = (
        proc_summary.groupby("categoría")
        .agg(monto=("monto_total_crc", "sum"), n=("monto_total_crc", "count"))
        .reset_index()
        .sort_values("monto", ascending=False)
    )
    fig_cat = px.bar(
        cat_data, y="categoría", x="monto", orientation="h",
        title="Monto por Categoría de Producto",
        labels={"monto": "Monto (₡)", "categoría": ""},
        color="monto", color_continuous_scale="Viridis",
    )
    fig_cat.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(400, len(cat_data) * 35),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with col_cat_table:
    cat_tbl = cat_data.copy()
    cat_tbl["pct"] = (cat_tbl["monto"] / cat_tbl["monto"].sum() * 100).round(1)
    cat_tbl.columns = ["Categoría", "Monto (₡)", "Procedimientos", "% del Total"]
    cat_tbl["Monto (₡)"] = cat_tbl["Monto (₡)"].apply(fmt_crc)
    cat_tbl.index = range(1, len(cat_tbl) + 1)
    st.dataframe(cat_tbl, width="stretch", hide_index=True)

    csv_cat = cat_data.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar categorías (CSV)", csv_cat,
                       "categorias_producto.csv", "text/csv")

st.markdown("---")

# ── recursos / appeals analysis ──
st.subheader("⚖️ Análisis de Recursos (Apelaciones)")
col_rec_pie, col_rec_year = st.columns(2)

with col_rec_pie:
    rec = proc_summary.copy()
    rec["permite"] = rec["permite recursos"].fillna("Sin dato").str.strip()
    rec["permite"] = rec["permite"].replace({"": "Sin dato"})
    rec_cnt = rec["permite"].value_counts().reset_index()
    rec_cnt.columns = ["Permite Recursos", "Cantidad"]
    fig_rec = px.pie(
        rec_cnt, names="Permite Recursos", values="Cantidad",
        title="¿Permite Recursos?",
        color="Permite Recursos",
        color_discrete_map={"Si": "#e67e22", "No": "#2ecc71", "Sin dato": "#95a5a6"},
    )
    st.plotly_chart(fig_rec, use_container_width=True)

with col_rec_year:
    rec_y = df[df["fecha_adj"].notna()].copy()
    rec_y["año"] = rec_y["fecha_adj"].dt.year.astype(int)
    rec_y["permite"] = rec_y["permite recursos"].fillna("Sin dato").str.strip()
    rec_y["permite"] = rec_y["permite"].replace({"": "Sin dato"})
    rec_yearly = (
        rec_y.drop_duplicates("número de procedimiento")
        .groupby(["año", "permite"]).size().reset_index(name="n")
    )
    fig_ry = px.bar(rec_yearly, x="año", y="n", color="permite",
                    title="Permite Recursos por Año", barmode="stack",
                    color_discrete_map={"Si": "#e67e22", "No": "#2ecc71", "Sin dato": "#95a5a6"},
                    labels={"año": "", "n": "Procedimientos", "permite": "Permite Recursos"})
    st.plotly_chart(fig_ry, use_container_width=True)

st.markdown("---")

# ── concentration (top 10 proveedores share) ──
st.subheader("🔬 Concentración del Gasto")
if len(prov_rank) >= 2:
    prov_rank_all = prov_rank.copy()
    prov_rank_all["pct"] = prov_rank_all["monto_total"] / prov_rank_all["monto_total"].sum() * 100
    prov_rank_all["pct_acum"] = prov_rank_all["pct"].cumsum()

    top10_share = prov_rank_all.head(10)["pct"].sum()
    top5_share  = prov_rank_all.head(5)["pct"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Top 5 proveedores", f"{top5_share:.1f}% del gasto")
    c2.metric("Top 10 proveedores", f"{top10_share:.1f}% del gasto")
    c3.metric("Total proveedores con montos", f"{len(prov_rank_all):,}")

    fig_conc = px.line(
        prov_rank_all.head(30), x="nombre proveedor", y="pct_acum",
        title="Curva de concentración (primeros 30 proveedores)",
        labels={"nombre proveedor": "", "pct_acum": "% acumulado del gasto"},
        markers=True,
    )
    fig_conc.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig_conc, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("🚨 Anomalías Detectadas")
st.caption("Procedimientos con montos estadísticamente atípicos o patrones inusuales")

with_amounts = proc_summary[proc_summary["monto_total_crc"] > 0].copy()

if len(with_amounts) >= 10:
    mean_amt = with_amounts["monto_total_crc"].mean()
    std_amt  = with_amounts["monto_total_crc"].std()
    threshold_high = mean_amt + 2 * std_amt

    type_stats = with_amounts.groupby("tipo procedimiento")["monto_total_crc"].agg(["mean", "std"]).reset_index()
    type_stats.columns = ["tipo procedimiento", "type_mean", "type_std"]
    with_amounts = with_amounts.merge(type_stats, on="tipo procedimiento", how="left")
    with_amounts["type_std"] = with_amounts["type_std"].fillna(0)

    flags = []
    for _, row in with_amounts.iterrows():
        reasons = []
        amt = row["monto_total_crc"]
        if amt > threshold_high:
            reasons.append(f"Monto excede 2σ global (>{fmt_crc(threshold_high)})")
        if row["type_std"] > 0 and amt > row["type_mean"] + 2 * row["type_std"]:
            reasons.append(f"Monto excede 2σ para {row['tipo procedimiento']}")
        if row["type_std"] > 0 and amt < row["type_mean"] - 1.5 * row["type_std"] and amt > 0:
            reasons.append("Monto inusualmente bajo para su tipo")
        flags.append(" | ".join(reasons) if reasons else "")
    with_amounts["anomalía"] = flags

    anomalies = with_amounts[with_amounts["anomalía"] != ""].sort_values("monto_total_crc", ascending=False)

    col_anom_kpi1, col_anom_kpi2, col_anom_kpi3 = st.columns(3)
    col_anom_kpi1.metric("Anomalías detectadas", f"{len(anomalies):,}")
    col_anom_kpi2.metric("Umbral global 2σ", fmt_crc(threshold_high))
    col_anom_kpi3.metric("% del total", f"{len(anomalies)/len(with_amounts)*100:.1f}%")

    with_amounts["es_anomalía"] = with_amounts["anomalía"] != ""
    fig_anom = px.scatter(
        with_amounts, x="fecha_adj", y="monto_total_crc",
        color="es_anomalía",
        color_discrete_map={True: "#e74c3c", False: "#95a5a6"},
        hover_data=["número de procedimiento", "descripción", "proveedores"],
        title="Monto por Procedimiento (anomalías en rojo)",
        labels={"fecha_adj": "Fecha", "monto_total_crc": "Monto (₡)",
                "es_anomalía": "Anomalía"},
    )
    fig_anom.add_hline(y=threshold_high, line_dash="dash", line_color="red",
                       annotation_text=f"Umbral 2σ: {fmt_crc(threshold_high)}")
    fig_anom.add_hline(y=mean_amt, line_dash="dot", line_color="blue",
                       annotation_text=f"Promedio: {fmt_crc(mean_amt)}")
    fig_anom.update_layout(height=500)
    st.plotly_chart(fig_anom, use_container_width=True)

    if len(anomalies) > 0:
        anom_disp = anomalies[[
            "número de procedimiento", "descripción", "proveedores",
            "monto_total_crc", "tipo procedimiento", "fecha_adj", "anomalía",
        ]].copy()
        anom_disp.columns = ["Procedimiento", "Descripción", "Proveedor(es)",
                             "Monto (₡)", "Tipo", "Fecha", "Razón"]
        anom_disp["Monto (₡)"] = anom_disp["Monto (₡)"].apply(fmt_crc)
        anom_disp["Fecha"] = pd.to_datetime(anom_disp["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
        anom_disp.index = range(1, len(anom_disp) + 1)
        st.dataframe(anom_disp, width="stretch", height=500)

        csv_anom = anom_disp.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar anomalías (CSV)", csv_anom,
                           "anomalias_adjudicaciones.csv", "text/csv")
    else:
        st.info("No se detectaron anomalías con los filtros actuales.")
else:
    st.info("Datos insuficientes para detectar anomalías.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FULL DETAIL TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("📑 Detalle Completo")

detail = proc_summary[[
    "número de procedimiento", "descripción", "categoría", "proveedores",
    "monto_total_crc", "tipo procedimiento", "modalidad procedimiento",
    "fecha_adj", "desierto", "permite recursos", "n_contratos",
]].copy()
detail.columns = [
    "Procedimiento", "Descripción", "Categoría", "Proveedor(es)",
    "Monto (₡)", "Tipo", "Modalidad",
    "Fecha Adj. Firme", "Desierto", "Recursos", "Contratos",
]
detail = detail.sort_values("Monto (₡)", ascending=False)
detail["Monto (₡)"]       = detail["Monto (₡)"].apply(fmt_crc)
detail["Fecha Adj. Firme"] = pd.to_datetime(
    detail["Fecha Adj. Firme"], errors="coerce"
).dt.strftime("%Y-%m-%d")
detail.index = range(1, len(detail) + 1)

st.dataframe(detail, width="stretch", height=600)

csv_all = detail.to_csv(index=False).encode("utf-8")
st.download_button("📥 Descargar todo (CSV)", csv_all,
                   "adjudicaciones_completo.csv", "text/csv")
