"""
Observador de Gasto Público (SICOP)
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
    page_title="Observador de Gasto Público (SICOP)",
    page_icon="🏛️",
    layout="wide",
)

st.markdown("""
<style>
    /* Tab labels — broad selectors for compatibility */
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"] {
        padding: 0.6rem 1.2rem !important;
        color: #fafafa !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0068c9 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span {
        color: #0068c9 !important;
    }
    /* Streamlit >=1.37 uses stTabs */
    .stTabs [data-baseweb="tab-list"] button p,
    .stTabs [data-baseweb="tab-list"] button span,
    .stTabs [data-baseweb="tab-list"] button div {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        padding: 0.6rem 1.2rem !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p,
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] span,
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] div {
        color: #0068c9 !important;
    }
    /* Sidebar filter labels */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

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
    "Municipalidad de Montes de Oca (3014042053)": "3014042053",
    "Municipalidad de San José (3014042058)": "3014042058",
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
    dp["fecha_publicacion"] = pd.to_datetime(dp["fecha publicación"], errors="coerce")
    dp["fecha_cierre"]      = pd.to_datetime(dp["fecha cierre recepción"], errors="coerce")
    dp_uniq = (
        dp.sort_values("número procedimiento")
        .drop_duplicates(subset="número procedimiento")
        [["número procedimiento", "descripción", "categoría", "tipo procedimiento",
          "modalidad procedimiento", "nombre unidad compra",
          "fecha_publicacion", "fecha_cierre"]]
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

    # ── 5. vigencia from contratos (contract duration) ──
    con["vigencia_num"] = pd.to_numeric(con["vigencia contrato"], errors="coerce")
    con["unidad_vig"]   = con.get("unidad vigencia", pd.Series(dtype="object")).fillna("").str.strip()
    con["vigencia_dias"] = np.where(
        con["unidad_vig"].str.lower().str.startswith("año"), con["vigencia_num"] * 365,
        np.where(con["unidad_vig"].str.lower().str.startswith("mes"), con["vigencia_num"] * 30, np.nan)
    )
    vig_proc = (
        con.groupby("nro procedimiento")
        .agg(vigencia_dias=("vigencia_dias", "mean"))
        .reset_index()
        .rename(columns={"nro procedimiento": "_proc"})
    )
    enriched = enriched.merge(vig_proc, left_on="número de procedimiento",
                              right_on="_proc", how="left").drop(columns=["_proc"], errors="ignore")

    # ── 6. compute plazo columns ──
    enriched["plazo_proceso_dias"] = (enriched["fecha_adj"] - enriched["fecha_publicacion"]).dt.days
    enriched["ventana_ofertas_dias"] = (enriched["fecha_cierre"] - enriched["fecha_publicacion"]).dt.days

    # ── 6b. annualized amount (monto / (vigencia_dias / 360)) ──
    enriched["monto_anualizado"] = np.where(
        enriched["vigencia_dias"] > 0,
        enriched["monto_total_crc"] / (enriched["vigencia_dias"] / 360),
        np.nan,
    )

    # ── 7. proveedor‑level amounts (for top‑proveedor ranking) ──
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

st.sidebar.title("🏛️ Observador de Gasto Público (SICOP)")
st.sidebar.markdown("---")

sel_inst = st.sidebar.selectbox("Institución", list(INSTITUTIONS.keys()))
cedula = INSTITUTIONS[sel_inst]
inst_label = sel_inst.split(" (")[0]

tables = load_tables(cedula)
enriched, op_con, op_raw = prepare_data(tables)

# year filter
valid_dates = enriched["fecha_adj"].dropna()
if len(valid_dates):
    all_years = sorted(y for y in valid_dates.dt.year.unique() if y >= 2021)
    sel_years = st.sidebar.multiselect("Año(s)", all_years, default=all_years)
else:
    sel_years = []

# categoría filter
categorias = sorted(enriched["categoría"].dropna().unique())
sel_categorias = st.sidebar.multiselect("Categoría de Producto", categorias,
                                         placeholder="Todos")

# tipo procedimiento
tipos = sorted(enriched["tipo procedimiento"].dropna().unique())
sel_tipos = st.sidebar.multiselect("Tipo de Procedimiento", tipos,
                                    placeholder="Todos")

# ── apply filters ──
df = enriched.copy()

if sel_years:
    df = df[df["fecha_adj"].dt.year.isin(sel_years) | df["fecha_adj"].isna()]

if sel_tipos:
    df = df[df["tipo procedimiento"].isin(sel_tipos) | df["tipo procedimiento"].isna()]

if sel_categorias:
    df = df[df["categoría"].isin(sel_categorias) | df["categoría"].isna()]

# filter op_con to match filtered procedures (all visualizations use this)
filtered_procs = set(df["número de procedimiento"].dropna().unique())
op_filtered = op_con[op_con["número procedimiento"].isin(filtered_procs)]

# add category to op_filtered via procedure mapping
proc_cat = df.drop_duplicates("número de procedimiento")[["número de procedimiento", "categoría"]]
op_filtered = op_filtered.merge(
    proc_cat, left_on="número procedimiento",
    right_on="número de procedimiento", how="left",
).drop(columns=["número de procedimiento"], errors="ignore")

# ── view selector ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
vista = st.sidebar.radio("📊 Vista", ["Adjudicaciones", "Análisis de Oferentes"],
                         index=0)


# ── helpers ───────────────────────────────────────────────────────────────────

def fmt_crc(v):
    return f"₡{v:,.0f}" if pd.notna(v) and v != 0 else "—"

def monto_col(use_anual: bool) -> str:
    """Return the column name to use for amounts."""
    return "monto_anualizado" if use_anual else "monto_total_crc"

def monto_label(use_anual: bool) -> str:
    """Return the display label for amounts."""
    return "Monto Anualizado (₡)" if use_anual else "Monto (₡)"

# ── common data prep (shared across tabs) ─────────────────────────────────────

proc_dedup = df.drop_duplicates(subset="número de procedimiento")
n_procs    = len(proc_dedup)
total_monto = proc_dedup["monto_total_crc"].sum()
with_monto  = (proc_dedup["monto_total_crc"] > 0).sum()
avg_monto   = proc_dedup.loc[proc_dedup["monto_total_crc"] > 0, "monto_total_crc"].mean() if with_monto else 0
n_prov      = (proc_dedup["proveedores"].dropna()
               .str.split(r"\s*\|\s*").explode().nunique())
n_desierto  = (proc_dedup["desierto"].str.strip().str.upper() == "S").sum()
proc_summary = proc_dedup.copy()

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


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

st.title(f"🏛️ Observador de Gasto Público (SICOP) — {inst_label}")
st.caption(f"**{n_procs:,}** procedimientos · {len(df):,} registros de adjudicación")

tab_home, tab_gasto, tab_anomalias, tab_oferentes, tab_legal, tab_glosario = st.tabs(
    ["🏠 Inicio", "💰 Gasto", "🚨 Anomalías", "🔎 Oferentes",
     "⚖️ Correlaciones Legales", "📖 Glosario"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB: INICIO (Landing Page)
# ══════════════════════════════════════════════════════════════════════════════

with tab_home:
    st.header("Bienvenido al Observador de Gasto Público (SICOP)")

    anual_home = st.toggle("Mostrar montos anualizados", value=False, key="anual_home")
    _monto_home = proc_dedup[monto_col(anual_home)].fillna(0)
    _total_home = _monto_home.sum()

    col_kpi, col_donut = st.columns([2, 3])

    with col_kpi:
        k1, k2 = st.columns(2)
        k1.metric("Procedimientos", f"{n_procs:,}")
        k2.metric(monto_label(anual_home), fmt_crc(_total_home))
        k3, k4 = st.columns(2)
        k3.metric("Proveedores", f"{n_prov:,}")
        k4.metric("Desiertos", f"{n_desierto:,}")

    with col_donut:
        _mc_home = monto_col(anual_home)
        cat_donut = (
            proc_dedup.groupby("categoría")[_mc_home]
            .sum().reset_index()
            .sort_values(_mc_home, ascending=False)
        )
        cat_donut.columns = ["Categoría", "Monto"]
        fig_donut = px.pie(
            cat_donut, names="Categoría", values="Monto",
            title="Distribución del Gasto por Categoría",
            hole=0.45,
        )
        fig_donut.update_layout(
            height=500,
            margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(font=dict(size=10), orientation="h", yanchor="top", y=-0.05),
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent",
                                textfont_size=9)
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    st.markdown(f"""
    ### 🚀 Para comenzar
    Utilice los filtros de la **barra lateral izquierda** para acotar los datos:
    1. **Institución** — seleccione la entidad pública a analizar
    2. **Año(s)** — escoja uno o varios períodos fiscales
    3. **Tipo de Procedimiento** — filtre por tipo de gasto (Licitación Pública,
       Contratación Directa, Licitación Abreviada, etc.)

    También puede refinar por proveedor, categoría de producto o estado
    (adjudicado / desierto).

    > Actualmente visualizando: **{inst_label}**
    """)

    st.markdown("---")

    lc1, lc2, lc3 = st.columns(3)

    with lc1:
        st.subheader("💰 Gasto")
        st.markdown("""
        Visión integral del gasto público adjudicado:
        - **Top Proveedores** por monto adjudicado
        - **Top 50** adjudicaciones de mayor monto
        - **Análisis Temporal** mensual y anual
        - **Distribución** por tipo y modalidad de procedimiento
        - **Categorías de producto** con clasificación automática
        - **Recursos / Apelaciones** permitidas
        - **Concentración** del gasto entre proveedores
        - **Plazos** del proceso, ventana de ofertas y vigencia contractual
        - **Tabla de detalle completo** con descarga CSV
        """)

    with lc2:
        st.subheader("🚨 Anomalías")
        st.markdown("""
        Detección automática de procedimientos atípicos:
        - **Outliers estadísticos**: montos que exceden 2σ del promedio global
        - **Outliers por tipo**: montos atípicos dentro de cada tipo de procedimiento
        - **Montos inusualmente bajos** para su categoría
        - **Gráfico scatter** con umbral y promedio superpuestos
        - **Tabla descargable** de todos los procedimientos anómalos con
          la razón específica de cada alerta
        """)

    with lc3:
        st.subheader("🔎 Oferentes")
        st.markdown("""
        Análisis detallado de los proveedores adjudicados:
        - **Treemap** de gasto por categoría y oferente
        - **Top 5 oferentes** por cada categoría de producto
        - **Concentración por categoría**: dominancia del Top 1 y Top 3
        - **Oferentes multi-categoría**: proveedores diversificados
        - **Drill-down** individual por oferente (gráfico + tabla)
        - **Tabla completa** de oferentes con descarga CSV
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: GASTO
# ══════════════════════════════════════════════════════════════════════════════

with tab_gasto:

    # ── TEMPORAL ──
    st.subheader("📈 Análisis Temporal")
    st.caption("Evolución del gasto adjudicado por mes y cantidad de procedimientos por año")
    anual_temporal = st.toggle("Mostrar montos anualizados", value=False, key="anual_temporal")
    _mc_t = monto_col(anual_temporal)
    col_monthly, col_yearly = st.columns(2)

    with col_monthly:
        dft = df[df["fecha_adj"].notna()].copy()
        dft["mes"] = dft["fecha_adj"].dt.to_period("M").astype(str)
        monthly = (
            dft.drop_duplicates("número de procedimiento")
            .groupby("mes")
            .agg(monto=(_mc_t, "sum"), n=(_mc_t, "count"))
            .reset_index()
        )
        fig_m = px.bar(monthly, x="mes", y="monto",
                       title="Monto Adjudicado por Mes",
                       labels={"mes": "", "monto": monto_label(anual_temporal)})
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
                       labels={"año": "", "n": "Procedimientos"}, text="n")
        fig_y.update_traces(textposition="outside")
        st.plotly_chart(fig_y, use_container_width=True)

    st.markdown("---")

    # ── CATEGORÍA ──
    st.subheader("🏷️ Distribución por Categoría de Producto")
    st.caption("Clasificación automática del gasto por categoría de bien o servicio")
    anual_cat = st.toggle("Mostrar montos anualizados", value=False, key="anual_cat")
    _mc_cat = monto_col(anual_cat)
    _ml_cat = monto_label(anual_cat)
    col_cat_chart, col_cat_table = st.columns([3, 2])

    with col_cat_chart:
        cat_data = (
            proc_summary.groupby("categoría")
            .agg(monto=(_mc_cat, "sum"), n=(_mc_cat, "count"))
            .reset_index().sort_values("monto", ascending=False)
        )
        cat_data["pct"] = (cat_data["monto"] / cat_data["monto"].sum() * 100).round(1)
        cat_data["pct_label"] = cat_data["pct"].apply(lambda x: f"{x:.1f}%")
        fig_cat = px.bar(
            cat_data, y="categoría", x="monto", orientation="h",
            title="Monto por Categoría de Producto",
            labels={"monto": _ml_cat, "categoría": ""},
            color="monto", color_continuous_scale="Viridis",
            text="pct_label",
        )
        fig_cat.update_traces(textposition="outside", textfont_size=11)
        fig_cat.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=max(400, len(cat_data) * 35), coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_cat_table:
        cat_tbl = cat_data[["categoría", "monto", "n"]].copy()
        cat_tbl["pct"] = (cat_tbl["monto"] / cat_tbl["monto"].sum() * 100).round(1)
        cat_tbl.columns = ["Categoría", _ml_cat, "Procedimientos", "% del Total"]
        cat_tbl[_ml_cat] = cat_tbl[_ml_cat].apply(fmt_crc)
        cat_tbl.index = range(1, len(cat_tbl) + 1)
        st.dataframe(cat_tbl, width="stretch", hide_index=True)
        csv_cat = cat_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar categorías (CSV)", csv_cat,
                           "categorias_producto.csv", "text/csv")

    st.markdown("---")

    # ── TIPO / MODALIDAD ──
    st.subheader("📋 Distribución por Tipo de Procedimiento")
    st.caption("Composición del gasto según el tipo de contratación")
    anual_tipo = st.toggle("Mostrar montos anualizados", value=False, key="anual_tipo")
    _mc_tp = monto_col(anual_tipo)
    _ml_tp = monto_label(anual_tipo)

    tipo_data = (
        proc_summary.groupby("tipo procedimiento")
        .agg(monto=(_mc_tp, "sum"), n=(_mc_tp, "count"))
        .reset_index().sort_values("monto", ascending=False)
    )
    col_tipo_chart, col_tipo_table = st.columns([3, 2])
    with col_tipo_chart:
        fig_t = px.pie(tipo_data, names="tipo procedimiento", values="monto",
                       title="Monto por Tipo de Procedimiento")
        st.plotly_chart(fig_t, use_container_width=True)
    with col_tipo_table:
        tipo_tbl = tipo_data.copy()
        tipo_tbl.columns = ["Tipo Procedimiento", _ml_tp, "Cantidad"]
        tipo_tbl[_ml_tp] = tipo_tbl[_ml_tp].apply(fmt_crc)
        st.dataframe(tipo_tbl, width="stretch", hide_index=True)

    st.markdown("---")

    # ── TOP PROVEEDORES ──
    st.subheader("🏆 Top Proveedores por Monto Adjudicado")
    st.caption("Proveedores con mayor monto total adjudicado en el período seleccionado")

    top_n = st.slider("Proveedores a mostrar", 10, 50, 25, key="prov_slider")
    top_prov = prov_rank.head(top_n).copy()

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        fig_prov = px.bar(
            top_prov, y="nombre proveedor", x="monto_total", orientation="h",
            title=f"Top {top_n} Proveedores (₡)",
            labels={"monto_total": "Monto (₡)", "nombre proveedor": ""},
            color="monto_total", color_continuous_scale="Blues",
        )
        fig_prov.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=max(450, top_n * 28), coloraxis_showscale=False,
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
        st.dataframe(tbl_prov, width="stretch", height=max(450, top_n * 28))

        csv_prov = prov_rank.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar lista completa proveedores",
                           csv_prov, "proveedores_por_monto.csv", "text/csv")

    st.markdown("---")

    # ── CONCENTRACIÓN ──
    st.subheader("🔬 Concentración del Gasto")
    st.caption("Distribución del gasto entre proveedores y nivel de concentración en los principales")
    if len(prov_rank) >= 2:
        prov_rank_all = prov_rank.copy()
        prov_rank_all["pct"] = prov_rank_all["monto_total"] / prov_rank_all["monto_total"].sum() * 100
        prov_rank_all["pct_acum"] = prov_rank_all["pct"].cumsum()

        top10_share = prov_rank_all.head(10)["pct"].sum()
        top5_share  = prov_rank_all.head(5)["pct"].sum()

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Top 5 proveedores", f"{top5_share:.1f}% del gasto")
        cc2.metric("Top 10 proveedores", f"{top10_share:.1f}% del gasto")
        cc3.metric("Total proveedores con montos", f"{len(prov_rank_all):,}")

        fig_conc = px.line(
            prov_rank_all.head(30), x="nombre proveedor", y="pct_acum",
            title="Curva de concentración (primeros 30 proveedores)",
            labels={"nombre proveedor": "", "pct_acum": "% acumulado del gasto"},
            markers=True,
        )
        fig_conc.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_conc, use_container_width=True)

    st.markdown("---")

    # ── PLAZOS ──
    st.subheader("⏱️ Análisis de Plazos (en días)")
    st.caption("Duración del proceso, ventana de ofertas y vigencia contractual")

    plazo_df = proc_summary.copy()
    has_proceso  = plazo_df["plazo_proceso_dias"].notna() & (plazo_df["plazo_proceso_dias"] > 0)
    has_ventana  = plazo_df["ventana_ofertas_dias"].notna() & (plazo_df["ventana_ofertas_dias"] > 0)
    has_vigencia = plazo_df["vigencia_dias"].notna() & (plazo_df["vigencia_dias"] > 0)

    pk1, pk2, pk3 = st.columns(3)
    pk1.metric("Plazo promedio del proceso",
               f"{plazo_df.loc[has_proceso, 'plazo_proceso_dias'].mean():.0f} días"
               if has_proceso.sum() else "—")
    pk2.metric("Ventana promedio de ofertas",
               f"{plazo_df.loc[has_ventana, 'ventana_ofertas_dias'].mean():.0f} días"
               if has_ventana.sum() else "—")
    pk3.metric("Vigencia promedio del contrato",
               f"{plazo_df.loc[has_vigencia, 'vigencia_dias'].mean():.0f} días"
               if has_vigencia.sum() else "—")

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        data_proc = plazo_df.loc[has_proceso, "plazo_proceso_dias"]
        if len(data_proc) > 0:
            fig_pp = px.histogram(
                data_proc, nbins=40,
                title="Duración del Proceso (días)",
                labels={"value": "Días", "count": "Procedimientos"},
                color_discrete_sequence=["#3498db"],
            )
            fig_pp.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_pp, use_container_width=True)
        else:
            st.info("Sin datos de plazo del proceso.")

    with col_p2:
        data_vent = plazo_df.loc[has_ventana, "ventana_ofertas_dias"]
        if len(data_vent) > 0:
            fig_pv = px.histogram(
                data_vent, nbins=40,
                title="Ventana de Ofertas (días)",
                labels={"value": "Días", "count": "Procedimientos"},
                color_discrete_sequence=["#2ecc71"],
            )
            fig_pv.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_pv, use_container_width=True)
        else:
            st.info("Sin datos de ventana de ofertas.")

    with col_p3:
        data_vig = plazo_df.loc[has_vigencia, "vigencia_dias"]
        if len(data_vig) > 0:
            fig_pvg = px.histogram(
                data_vig, nbins=40,
                title="Vigencia del Contrato (días)",
                labels={"value": "Días", "count": "Procedimientos"},
                color_discrete_sequence=["#e67e22"],
            )
            fig_pvg.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_pvg, use_container_width=True)
        else:
            st.info("Sin datos de vigencia contractual.")

    plazo_scatter = plazo_df[has_proceso & (plazo_df["monto_total_crc"] > 0)].copy()
    if len(plazo_scatter) >= 5:
        fig_ps = px.scatter(
            plazo_scatter, x="plazo_proceso_dias", y="monto_total_crc",
            color="tipo procedimiento",
            hover_data=["número de procedimiento", "descripción"],
            title="Plazo del Proceso vs Monto Adjudicado",
            labels={"plazo_proceso_dias": "Plazo del Proceso (días)",
                    "monto_total_crc": "Monto (₡)"},
        )
        fig_ps.update_layout(height=450)
        st.plotly_chart(fig_ps, use_container_width=True)

    plazo_tipo = plazo_df[has_proceso].copy()
    if len(plazo_tipo) >= 5:
        fig_pb = px.box(
            plazo_tipo, x="tipo procedimiento", y="plazo_proceso_dias",
            title="Plazo del Proceso por Tipo de Procedimiento",
            labels={"tipo procedimiento": "", "plazo_proceso_dias": "Días"},
            color="tipo procedimiento",
        )
        fig_pb.update_layout(height=400, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_pb, use_container_width=True)

    plazo_detail = plazo_df[has_proceso | has_ventana | has_vigencia][[
        "número de procedimiento", "descripción", "tipo procedimiento",
        "plazo_proceso_dias", "ventana_ofertas_dias", "vigencia_dias",
        "monto_total_crc",
    ]].copy()
    plazo_detail.columns = ["Procedimiento", "Descripción", "Tipo",
                            "Plazo Proceso (días)", "Ventana Ofertas (días)",
                            "Vigencia Contrato (días)", "Monto (₡)"]
    plazo_detail = plazo_detail.sort_values("Plazo Proceso (días)", ascending=False)
    plazo_detail["Monto (₡)"] = plazo_detail["Monto (₡)"].apply(fmt_crc)
    plazo_detail.index = range(1, len(plazo_detail) + 1)
    st.dataframe(plazo_detail, width="stretch", height=400)

    csv_plazo = plazo_detail.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar plazos (CSV)", csv_plazo,
                       "plazos_adjudicaciones.csv", "text/csv")

    st.markdown("---")

    # ── ADJUDICACIONES MÁS RÁPIDAS QUE EL PROMEDIO ──
    st.subheader("⚡ Adjudicaciones Más Rápidas que el Promedio por Tipo")
    st.caption("Procedimientos cuyo plazo fue inferior al promedio de su tipo de procedimiento")

    rapidas_df = proc_summary[
        proc_summary["plazo_proceso_dias"].notna()
        & (proc_summary["plazo_proceso_dias"] > 0)
        & (proc_summary["monto_total_crc"] > 0)
    ].copy()

    if len(rapidas_df) >= 5:
        tipo_avg = (
            rapidas_df.groupby("tipo procedimiento")["plazo_proceso_dias"]
            .agg(promedio="mean", mediana="median", n="count")
            .reset_index()
        )
        rapidas_df = rapidas_df.merge(
            tipo_avg[["tipo procedimiento", "promedio"]],
            on="tipo procedimiento", how="left",
        )
        rapidas_df["días_bajo_promedio"] = rapidas_df["promedio"] - rapidas_df["plazo_proceso_dias"]
        rapidas_df["pct_bajo_promedio"] = (
            (rapidas_df["días_bajo_promedio"] / rapidas_df["promedio"]) * 100
        ).round(1)

        rapidas = rapidas_df[rapidas_df["días_bajo_promedio"] > 0].sort_values(
            "pct_bajo_promedio", ascending=False
        )

        rk1, rk2, rk3 = st.columns(3)
        rk1.metric("Más rápidas que promedio", f"{len(rapidas):,} de {len(rapidas_df):,}")
        rk2.metric("% más rápidas", f"{len(rapidas)/len(rapidas_df)*100:.1f}%")
        rk3.metric("Ahorro promedio de días",
                    f"{rapidas['días_bajo_promedio'].mean():.0f} días" if len(rapidas) else "—")

        st.markdown("##### Promedio de plazo por tipo de procedimiento")
        avg_disp = tipo_avg.copy()
        avg_disp.columns = ["Tipo Procedimiento", "Promedio (días)", "Mediana (días)", "Procedimientos"]
        avg_disp["Promedio (días)"] = avg_disp["Promedio (días)"].round(1)
        avg_disp["Mediana (días)"] = avg_disp["Mediana (días)"].round(1)
        st.dataframe(avg_disp, hide_index=True, use_container_width=True)

        if len(rapidas) > 0:
            st.markdown("##### Ranking de adjudicaciones más rápidas")
            anual_rapidas = st.toggle("Mostrar montos anualizados", value=False, key="anual_rapidas")
            _mc_r = monto_col(anual_rapidas)
            _ml_r = monto_label(anual_rapidas)

            rapidas_disp = rapidas[[
                "número de procedimiento", "descripción", "tipo procedimiento",
                "plazo_proceso_dias", "promedio", "días_bajo_promedio",
                "pct_bajo_promedio", _mc_r, "proveedores",
            ]].head(50).copy()
            rapidas_disp.columns = [
                "Procedimiento", "Descripción", "Tipo",
                "Plazo (días)", "Promedio Tipo (días)", "Días Bajo Promedio",
                "% Más Rápido", _ml_r, "Proveedor(es)",
            ]
            rapidas_disp["Promedio Tipo (días)"] = rapidas_disp["Promedio Tipo (días)"].round(1)
            rapidas_disp[_ml_r] = rapidas_disp[_ml_r].apply(fmt_crc)
            rapidas_disp.index = range(1, len(rapidas_disp) + 1)
            st.dataframe(rapidas_disp, width="stretch", height=600)

            csv_rapidas = rapidas_disp.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Descargar ranking rápidas (CSV)", csv_rapidas,
                               "adjudicaciones_rapidas_ranking.csv", "text/csv")

            fig_rapidas = px.scatter(
                rapidas.head(100),
                x="plazo_proceso_dias", y=_mc_r,
                color="tipo procedimiento",
                size="pct_bajo_promedio",
                hover_data=["número de procedimiento", "descripción",
                            "días_bajo_promedio", "proveedores"],
                title="Plazo vs Monto — Adjudicaciones Más Rápidas que el Promedio",
                labels={
                    "plazo_proceso_dias": "Plazo (días)",
                    _mc_r: _ml_r,
                    "tipo procedimiento": "Tipo",
                    "pct_bajo_promedio": "% Más Rápido",
                },
            )
            fig_rapidas.update_layout(height=500)
            st.plotly_chart(fig_rapidas, use_container_width=True)
    else:
        st.info("Datos insuficientes de plazos para este análisis.")

    st.markdown("---")

    # ── TOP 50 ──
    st.subheader("💰 Top 50 Adjudicaciones por Monto")
    st.caption("Los 50 procedimientos con mayor monto adjudicado")
    anual_top50 = st.toggle("Mostrar montos anualizados", value=False, key="anual_top50")
    _mc50 = monto_col(anual_top50)

    top50 = proc_summary[proc_summary[_mc50].fillna(0) > 0].sort_values(_mc50, ascending=False).head(50)
    top50_disp = top50[[
        "número de procedimiento", "descripción", "categoría", "proveedores",
        _mc50, "tipo procedimiento", "modalidad procedimiento",
        "fecha_adj", "n_contratos",
    ]].copy()
    top50_disp.columns = [
        "Procedimiento", "Descripción", "Categoría", "Proveedor(es)",
        monto_label(anual_top50), "Tipo", "Modalidad", "Fecha Adj. Firme", "Contratos",
    ]
    top50_disp[monto_label(anual_top50)] = top50_disp[monto_label(anual_top50)].apply(fmt_crc)
    top50_disp["Fecha Adj. Firme"] = pd.to_datetime(
        top50_disp["Fecha Adj. Firme"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    top50_disp.index = range(1, len(top50_disp) + 1)
    st.dataframe(top50_disp, width="stretch", height=1200)
    csv50 = top50_disp.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar Top 50 (CSV)", csv50,
                       "top50_adjudicaciones.csv", "text/csv")
    st.markdown("---")

    # ── DETALLE COMPLETO ──
    st.subheader("📑 Detalle Completo")
    st.caption("Tabla con todos los procedimientos y sus datos principales, exportable a CSV")
    anual_detail = st.toggle("Mostrar montos anualizados", value=False, key="anual_detail")
    _mc_d = monto_col(anual_detail)
    _ml_d = monto_label(anual_detail)

    detail = proc_summary[[
        "número de procedimiento", "descripción", "categoría", "proveedores",
        _mc_d, "tipo procedimiento", "modalidad procedimiento",
        "fecha_adj", "desierto", "permite recursos", "n_contratos",
    ]].copy()
    detail.columns = [
        "Procedimiento", "Descripción", "Categoría", "Proveedor(es)",
        _ml_d, "Tipo", "Modalidad",
        "Fecha Adj. Firme", "Desierto", "Recursos", "Contratos",
    ]
    detail = detail.sort_values(_ml_d, ascending=False)
    detail[_ml_d] = detail[_ml_d].apply(fmt_crc)
    detail["Fecha Adj. Firme"] = pd.to_datetime(
        detail["Fecha Adj. Firme"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    detail.index = range(1, len(detail) + 1)

    st.dataframe(detail, width="stretch", height=600)
    csv_all = detail.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar todo (CSV)", csv_all,
                       "adjudicaciones_completo.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: ANOMALÍAS
# ══════════════════════════════════════════════════════════════════════════════

with tab_anomalias:
    st.subheader("🚨 Anomalías Detectadas")
    st.caption("Procedimientos con montos estadísticamente atípicos o patrones inusuales")
    anual_anom = st.toggle("Mostrar montos anualizados", value=False, key="anual_anom")
    _mc_a = monto_col(anual_anom)
    _ml_a = monto_label(anual_anom)

    with_amounts = proc_summary[proc_summary[_mc_a].fillna(0) > 0].copy()

    if len(with_amounts) >= 10:
        mean_amt = with_amounts[_mc_a].mean()
        std_amt  = with_amounts[_mc_a].std()
        threshold_high = mean_amt + 2 * std_amt

        type_stats = with_amounts.groupby("tipo procedimiento")[_mc_a].agg(["mean", "std"]).reset_index()
        type_stats.columns = ["tipo procedimiento", "type_mean", "type_std"]
        with_amounts = with_amounts.merge(type_stats, on="tipo procedimiento", how="left")
        with_amounts["type_std"] = with_amounts["type_std"].fillna(0)

        flags = []
        for _, row in with_amounts.iterrows():
            reasons = []
            amt = row[_mc_a]
            if amt > threshold_high:
                reasons.append(f"Monto excede 2σ global (>{fmt_crc(threshold_high)})")
            if row["type_std"] > 0 and amt > row["type_mean"] + 2 * row["type_std"]:
                reasons.append(f"Monto excede 2σ para {row['tipo procedimiento']}")
            if row["type_std"] > 0 and amt < row["type_mean"] - 1.5 * row["type_std"] and amt > 0:
                reasons.append("Monto inusualmente bajo para su tipo")
            flags.append(" | ".join(reasons) if reasons else "")
        with_amounts["anomalía"] = flags

        anomalies = with_amounts[with_amounts["anomalía"] != ""].sort_values(_mc_a, ascending=False)

        col_anom_kpi1, col_anom_kpi2, col_anom_kpi3 = st.columns(3)
        col_anom_kpi1.metric("Anomalías detectadas", f"{len(anomalies):,}")
        col_anom_kpi2.metric("Umbral global 2σ", fmt_crc(threshold_high))
        col_anom_kpi3.metric("% del total", f"{len(anomalies)/len(with_amounts)*100:.1f}%")

        with_amounts["es_anomalía"] = with_amounts["anomalía"] != ""
        fig_anom = px.scatter(
            with_amounts, x="fecha_adj", y=_mc_a,
            color="es_anomalía",
            color_discrete_map={True: "#e74c3c", False: "#95a5a6"},
            hover_data=["número de procedimiento", "descripción", "proveedores"],
            title=f"{'Monto Anualizado' if anual_anom else 'Monto'} por Procedimiento (anomalías en rojo)",
            labels={"fecha_adj": "Fecha", _mc_a: _ml_a,
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
                _mc_a, "tipo procedimiento", "fecha_adj",
                "vigencia_dias", "anomalía",
            ]].copy()
            anom_disp.columns = ["Procedimiento", "Descripción", "Proveedor(es)",
                                 _ml_a, "Tipo", "Fecha",
                                 "Vigencia Contrato (días)", "Razón"]
            anom_disp[_ml_a] = anom_disp[_ml_a].apply(fmt_crc)
            anom_disp["Monto/Día (₡)"] = np.where(
                anom_disp["Vigencia Contrato (días)"].notna() & (anom_disp["Vigencia Contrato (días)"] > 0),
                anomalies[_mc_a] / anom_disp["Vigencia Contrato (días)"],
                np.nan,
            )
            anom_disp["Monto/Día (₡)"] = anom_disp["Monto/Día (₡)"].apply(
                lambda v: fmt_crc(v) if pd.notna(v) else "—"
            )
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

    # ── ANOMALÍAS DE PLAZOS ──
    st.subheader("⏱️ Anomalías de Plazos")
    st.caption("Procedimientos cuya duración es estadísticamente atípica para su tipo")

    plazo_anom = proc_summary[
        proc_summary["plazo_proceso_dias"].notna() & (proc_summary["plazo_proceso_dias"] > 0)
    ].copy()

    if len(plazo_anom) >= 10:
        plazo_type_stats = (
            plazo_anom.groupby("tipo procedimiento")["plazo_proceso_dias"]
            .agg(["mean", "std", "median"]).reset_index()
        )
        plazo_type_stats.columns = ["tipo procedimiento", "plazo_mean", "plazo_std", "plazo_median"]
        plazo_anom = plazo_anom.merge(plazo_type_stats, on="tipo procedimiento", how="left")
        plazo_anom["plazo_std"] = plazo_anom["plazo_std"].fillna(0)

        global_mean = plazo_anom["plazo_proceso_dias"].mean()
        global_std  = plazo_anom["plazo_proceso_dias"].std()

        plazo_flags = []
        for _, row in plazo_anom.iterrows():
            reasons = []
            d = row["plazo_proceso_dias"]
            if d > global_mean + 2 * global_std:
                reasons.append(f"Plazo excede 2σ global (>{global_mean + 2*global_std:.0f} días)")
            if row["plazo_std"] > 0 and d > row["plazo_mean"] + 2 * row["plazo_std"]:
                reasons.append(f"Plazo excede 2σ para {row['tipo procedimiento']}")
            if row["plazo_std"] > 0 and d < row["plazo_mean"] - 1.5 * row["plazo_std"] and d > 0:
                reasons.append("Plazo inusualmente corto para su tipo")
            if d <= 3 and row["monto_total_crc"] > global_mean:
                reasons.append(f"Adjudicado en {d:.0f} días con monto alto")
            v = row.get("ventana_ofertas_dias")
            if pd.notna(v) and v > 0 and d > 0 and v / d > 0.9:
                reasons.append("Ventana de ofertas ≈ plazo total (adjudicación casi inmediata)")
            plazo_flags.append(" | ".join(reasons) if reasons else "")
        plazo_anom["anomalía_plazo"] = plazo_flags

        plazo_anomalies = plazo_anom[plazo_anom["anomalía_plazo"] != ""].sort_values(
            "plazo_proceso_dias", ascending=False
        )

        ak1, ak2, ak3 = st.columns(3)
        ak1.metric("Anomalías de plazo detectadas", f"{len(plazo_anomalies):,}")
        ak2.metric("Plazo promedio global", f"{global_mean:.0f} días")
        ak3.metric("Umbral global 2σ", f"{global_mean + 2*global_std:.0f} días")

        plazo_anom["es_anomalía_plazo"] = plazo_anom["anomalía_plazo"] != ""
        fig_plazo_anom = px.scatter(
            plazo_anom, x="plazo_proceso_dias", y="monto_total_crc",
            color="es_anomalía_plazo",
            color_discrete_map={True: "#e74c3c", False: "#95a5a6"},
            size="monto_total_crc",
            hover_data=["número de procedimiento", "descripción", "tipo procedimiento",
                        "ventana_ofertas_dias", "vigencia_dias"],
            title="Plazo del Proceso vs Monto (anomalías de plazo en rojo)",
            labels={"plazo_proceso_dias": "Plazo del Proceso (días)",
                    "monto_total_crc": "Monto (₡)",
                    "es_anomalía_plazo": "Anomalía"},
        )
        fig_plazo_anom.add_vline(
            x=global_mean + 2 * global_std, line_dash="dash", line_color="red",
            annotation_text=f"Umbral 2σ: {global_mean + 2*global_std:.0f}d"
        )
        fig_plazo_anom.add_vline(
            x=global_mean, line_dash="dot", line_color="blue",
            annotation_text=f"Promedio: {global_mean:.0f}d"
        )
        fig_plazo_anom.update_layout(height=550)
        fig_plazo_anom.update_traces(marker=dict(sizemin=4, sizeref=2.*max(plazo_anom["monto_total_crc"])/(40.**2)))
        st.plotly_chart(fig_plazo_anom, use_container_width=True)

        fig_box_plazo = px.box(
            plazo_anom, x="tipo procedimiento", y="plazo_proceso_dias",
            color="tipo procedimiento",
            points="outliers",
            title="Distribución de Plazos por Tipo (outliers visibles)",
            labels={"tipo procedimiento": "", "plazo_proceso_dias": "Días"},
        )
        fig_box_plazo.update_layout(height=400, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_box_plazo, use_container_width=True)

        fast_track = plazo_anom[
            (plazo_anom["plazo_proceso_dias"] <= 7) & (plazo_anom["monto_total_crc"] > 0)
        ].sort_values("monto_total_crc", ascending=False)

        if len(fast_track) > 0:
            st.markdown("---")
            st.subheader("⚡ Adjudicaciones Rápidas (≤ 7 días)")
            st.caption(f"{len(fast_track)} procedimientos adjudicados en 7 días o menos")

            fast_disp = fast_track[[
                "número de procedimiento", "descripción", "tipo procedimiento",
                "plazo_proceso_dias", "ventana_ofertas_dias", "monto_total_crc",
                "proveedores",
            ]].head(30).copy()
            fast_disp.columns = ["Procedimiento", "Descripción", "Tipo",
                                 "Plazo (días)", "Ventana Ofertas (días)",
                                 "Monto (₡)", "Proveedor(es)"]
            fast_disp["Monto (₡)"] = fast_disp["Monto (₡)"].apply(fmt_crc)
            fast_disp.index = range(1, len(fast_disp) + 1)
            st.dataframe(fast_disp, width="stretch", height=400)

        if len(plazo_anomalies) > 0:
            st.markdown("---")
            pa_disp = plazo_anomalies[[
                "número de procedimiento", "descripción", "tipo procedimiento",
                "plazo_proceso_dias", "ventana_ofertas_dias", "vigencia_dias",
                "monto_total_crc", "anomalía_plazo",
            ]].copy()
            pa_disp.columns = ["Procedimiento", "Descripción", "Tipo",
                               "Plazo Proceso (días)", "Ventana Ofertas (días)",
                               "Vigencia (días)", "Monto (₡)", "Razón"]
            pa_disp["Monto (₡)"] = pa_disp["Monto (₡)"].apply(fmt_crc)
            pa_disp.index = range(1, len(pa_disp) + 1)
            st.dataframe(pa_disp, width="stretch", height=500)

            csv_pa = pa_disp.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Descargar anomalías de plazo (CSV)", csv_pa,
                               "anomalias_plazos.csv", "text/csv")
    else:
        st.info("Datos insuficientes de plazos para detectar anomalías.")

    st.markdown("---")

    # ── CONCENTRACIÓN DE PROVEEDORES POR CATEGORÍA ──
    st.subheader("🏢 Concentración Anómala de Proveedores por Categoría")
    st.caption("Categorías donde un solo proveedor acapara una proporción desproporcionada del monto")

    op_cat = op_filtered[
        (op_filtered["monto_crc"] > 0) & op_filtered["categoría"].notna()
        & (op_filtered["categoría"] != "Sin categoría")
    ].copy()

    if len(op_cat) >= 20:
        cat_totals = op_cat.groupby("categoría").agg(
            monto_cat=("monto_crc", "sum"),
            n_proveedores=("cédula proveedor", "nunique"),
            n_contratos=("número procedimiento", "nunique"),
        ).reset_index()
        cat_totals = cat_totals[cat_totals["monto_cat"] > 0]

        prov_cat = op_cat.groupby(["categoría", "cédula proveedor", "nombre proveedor"]).agg(
            monto_prov=("monto_crc", "sum"),
            n_contratos_prov=("número procedimiento", "nunique"),
        ).reset_index()
        idx_top = prov_cat.groupby("categoría")["monto_prov"].idxmax()
        top_prov = prov_cat.loc[idx_top]

        conc = cat_totals.merge(
            top_prov[["categoría", "nombre proveedor", "cédula proveedor",
                       "monto_prov", "n_contratos_prov"]],
            on="categoría", how="left",
        )
        conc["pct_concentración"] = (conc["monto_prov"] / conc["monto_cat"] * 100).round(1)

        def calc_hhi(grp):
            shares = grp["monto_crc"] / grp["monto_crc"].sum() * 100
            return (shares ** 2).sum()
        hhi_cat = op_cat.groupby("categoría").apply(calc_hhi, include_groups=False).reset_index()
        hhi_cat.columns = ["categoría", "hhi"]
        conc = conc.merge(hhi_cat, on="categoría", how="left")

        conc["es_anomalía"] = (
            (conc["pct_concentración"] > 60) |
            (conc["hhi"] > 4000) |
            ((conc["n_proveedores"] == 1) & (conc["n_contratos"] >= 3))
        )
        conc = conc.sort_values("pct_concentración", ascending=False)

        anomalías_conc = conc[conc["es_anomalía"]]
        n_anom_conc = len(anomalías_conc)
        avg_conc = conc["pct_concentración"].mean()
        max_conc = conc["pct_concentración"].max()

        ck1, ck2, ck3 = st.columns(3)
        ck1.metric("Categorías con concentración anómala", f"{n_anom_conc}")
        ck2.metric("Concentración promedio del top proveedor", f"{avg_conc:.1f}%")
        ck3.metric("Mayor concentración", f"{max_conc:.1f}%")

        fig_conc_bar = px.bar(
            conc.sort_values("pct_concentración", ascending=True),
            x="pct_concentración", y="categoría",
            color="es_anomalía",
            color_discrete_map={True: "#e74c3c", False: "#3498db"},
            hover_data=["nombre proveedor", "monto_prov", "monto_cat",
                        "n_proveedores", "hhi"],
            title="% del Monto Concentrado en el Proveedor Principal (por Categoría)",
            labels={"pct_concentración": "Concentración (%)",
                    "categoría": "",
                    "es_anomalía": "Anomalía",
                    "nombre proveedor": "Proveedor principal",
                    "monto_prov": "Monto proveedor (₡)",
                    "monto_cat": "Monto categoría (₡)",
                    "n_proveedores": "# Proveedores",
                    "hhi": "HHI"},
            orientation="h",
        )
        fig_conc_bar.add_vline(x=60, line_dash="dash", line_color="red",
                               annotation_text="Umbral 60%")
        fig_conc_bar.update_layout(height=max(400, len(conc) * 35), showlegend=True)
        st.plotly_chart(fig_conc_bar, use_container_width=True)

        fig_hhi = px.scatter(
            conc, x="n_proveedores", y="hhi",
            size="monto_cat", color="es_anomalía",
            color_discrete_map={True: "#e74c3c", False: "#95a5a6"},
            hover_data=["categoría", "nombre proveedor", "pct_concentración"],
            title="Índice HHI vs Número de Proveedores por Categoría",
            labels={"n_proveedores": "# Proveedores en la Categoría",
                    "hhi": "Índice HHI",
                    "monto_cat": "Monto Total (₡)",
                    "es_anomalía": "Anomalía"},
        )
        fig_hhi.add_hline(y=4000, line_dash="dash", line_color="red",
                          annotation_text="HHI > 4000: Alta concentración")
        fig_hhi.add_hline(y=2500, line_dash="dot", line_color="orange",
                          annotation_text="HHI > 2500: Moderada")
        fig_hhi.update_layout(height=500)
        fig_hhi.update_traces(marker=dict(sizemin=6,
                              sizeref=2.*max(conc["monto_cat"])/(40.**2)))
        st.plotly_chart(fig_hhi, use_container_width=True)

        if n_anom_conc > 0:
            st.markdown("#### Detalle de Categorías con Concentración Anómala")
            conc_disp = anomalías_conc[[
                "categoría", "nombre proveedor", "pct_concentración",
                "monto_prov", "monto_cat", "n_proveedores", "n_contratos",
                "n_contratos_prov", "hhi",
            ]].copy()
            conc_disp.columns = [
                "Categoría", "Proveedor Principal", "Concentración (%)",
                "Monto Proveedor (₡)", "Monto Categoría (₡)",
                "# Proveedores", "# Contratos Total",
                "# Contratos Proveedor", "HHI",
            ]
            conc_disp["Monto Proveedor (₡)"] = conc_disp["Monto Proveedor (₡)"].apply(fmt_crc)
            conc_disp["Monto Categoría (₡)"] = conc_disp["Monto Categoría (₡)"].apply(fmt_crc)
            conc_disp["HHI"] = conc_disp["HHI"].round(0).astype(int)
            conc_disp.index = range(1, len(conc_disp) + 1)
            st.dataframe(conc_disp, width="stretch", height=400)

            csv_conc = conc_disp.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Descargar concentración anómala (CSV)", csv_conc,
                               "concentracion_anomala_categorias.csv", "text/csv")
    else:
        st.info("Datos insuficientes para analizar concentración por categoría.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: OFERENTES
# ══════════════════════════════════════════════════════════════════════════════

with tab_oferentes:

    op_ofer = op_filtered[op_filtered["monto_crc"] > 0].copy()
    op_ofer["nombre proveedor"] = op_ofer["nombre proveedor"].fillna(op_ofer["cédula proveedor"])
    op_ofer["categoría"] = op_ofer["categoría"].fillna("Sin categoría")

    prov_agg = (
        op_ofer.groupby(["cédula proveedor", "nombre proveedor", "tipo proveedor",
                         "tamaño proveedor", "provincia"])
        .agg(
            monto_total=("monto_crc", "sum"),
            n_ordenes=("monto_crc", "count"),
            n_procedimientos=("número procedimiento", "nunique"),
            n_categorias=("categoría", "nunique"),
        )
        .reset_index()
        .sort_values("monto_total", ascending=False)
    )

    prov_cat = (
        op_ofer.groupby(["nombre proveedor", "cédula proveedor", "categoría"])
        .agg(monto=("monto_crc", "sum"), ordenes=("monto_crc", "count"),
             procedimientos=("número procedimiento", "nunique"))
        .reset_index()
        .sort_values("monto", ascending=False)
    )

    total_ofer = prov_agg["cédula proveedor"].nunique()
    total_gasto = prov_agg["monto_total"].sum()
    avg_gasto = prov_agg["monto_total"].mean() if len(prov_agg) else 0
    cats_activas = op_ofer["categoría"].nunique()

    st.caption(f"Datos filtrados · **{total_ofer:,}** oferentes · **{cats_activas}** categorías")
    st.markdown("---")

    ok1, ok2, ok3, ok4 = st.columns(4)
    ok1.metric("Oferentes", f"{total_ofer:,}")
    ok2.metric("Gasto Total (₡)", fmt_crc(total_gasto))
    ok3.metric("Gasto Promedio / Oferente", fmt_crc(avg_gasto))
    ok4.metric("Categorías Activas", f"{cats_activas}")
    st.markdown("---")

    # ── 1. TREEMAP ──
    st.subheader("🗺️ Gasto por Oferente y Categoría")
    st.caption("Mapa de calor del gasto distribuido entre oferentes y categorías de producto")
    treemap_data = prov_cat[prov_cat["monto"] > 0].copy()
    treemap_top = treemap_data.sort_values("monto", ascending=False).head(100)

    fig_tree = px.treemap(
        treemap_top, path=["categoría", "nombre proveedor"], values="monto",
        title="Distribución del gasto: Categoría → Oferente (Top 100 combinaciones)",
        color="monto", color_continuous_scale="YlOrRd",
    )
    fig_tree.update_layout(height=600)
    st.plotly_chart(fig_tree, use_container_width=True)
    st.markdown("---")

    # ── 2. TOP OFERENTES POR CATEGORÍA ──
    st.subheader("🏅 Top Oferentes por Categoría")
    st.caption("Los 5 proveedores con mayor monto adjudicado en cada categoría de producto")
    cat_totals = prov_cat.groupby("categoría")["monto"].sum().reset_index()
    cat_totals.columns = ["categoría", "monto_cat_total"]
    prov_cat_pct = prov_cat.merge(cat_totals, on="categoría")
    prov_cat_pct["pct_cat"] = (prov_cat_pct["monto"] / prov_cat_pct["monto_cat_total"] * 100).round(1)

    top_per_cat = (
        prov_cat_pct.sort_values(["categoría", "monto"], ascending=[True, False])
        .groupby("categoría").head(5)
    )

    fig_top_cat = px.bar(
        top_per_cat, x="monto", y="nombre proveedor", color="categoría",
        facet_col="categoría", facet_col_wrap=3, orientation="h",
        title="Top 5 Oferentes por Categoría",
        labels={"monto": "Monto (₡)", "nombre proveedor": ""},
    )
    fig_top_cat.update_layout(
        height=max(500, (top_per_cat["categoría"].nunique() // 3 + 1) * 300),
        showlegend=False,
    )
    fig_top_cat.for_each_yaxis(lambda y: y.update(categoryorder="total ascending"))
    fig_top_cat.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1][:25]))
    st.plotly_chart(fig_top_cat, use_container_width=True)
    st.markdown("---")

    # ── 3. CONCENTRACIÓN POR CATEGORÍA ──
    st.subheader("📊 Concentración del Gasto por Categoría")
    st.caption("Porcentaje del gasto de cada categoría controlado por el Top 1 y Top 3 oferentes")

    conc_rows = []
    for cat, grp in prov_cat.groupby("categoría"):
        grp_sorted = grp.sort_values("monto", ascending=False)
        cat_total = grp_sorted["monto"].sum()
        if cat_total == 0:
            continue
        top1 = grp_sorted.head(1)["monto"].sum() / cat_total * 100
        top3 = grp_sorted.head(3)["monto"].sum() / cat_total * 100
        n_ofer = len(grp_sorted)
        lider = grp_sorted.iloc[0]["nombre proveedor"]
        conc_rows.append({
            "Categoría": cat, "Gasto Total": cat_total,
            "Top 1 (%)": round(top1, 1), "Top 3 (%)": round(top3, 1),
            "# Oferentes": n_ofer, "Líder": lider,
        })

    conc_df = pd.DataFrame(conc_rows).sort_values("Gasto Total", ascending=False)

    col_conc_chart, col_conc_tbl = st.columns([3, 2])
    with col_conc_chart:
        fig_conc_cat = px.bar(
            conc_df, y="Categoría", x=["Top 1 (%)", "Top 3 (%)"],
            orientation="h", barmode="overlay",
            title="Concentración: Top 1 vs Top 3 oferentes por categoría",
            labels={"value": "% del gasto", "Categoría": ""},
            color_discrete_sequence=["#e74c3c", "#3498db"],
        )
        fig_conc_cat.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=max(400, len(conc_df) * 35), legend_title_text="",
        )
        st.plotly_chart(fig_conc_cat, use_container_width=True)

    with col_conc_tbl:
        conc_disp = conc_df.copy()
        conc_disp["Gasto Total"] = conc_disp["Gasto Total"].apply(fmt_crc)
        conc_disp.index = range(1, len(conc_disp) + 1)
        st.dataframe(conc_disp, width="stretch", hide_index=True)

    st.markdown("---")

    # ── 4. MULTI-CATEGORÍA ──
    st.subheader("🔗 Oferentes Multi-Categoría")
    st.caption("Proveedores que participan en múltiples categorías de gasto")

    multi_cat = prov_agg[prov_agg["n_categorias"] >= 2].sort_values(
        "n_categorias", ascending=False
    ).head(30).copy()

    if len(multi_cat) > 0:
        fig_multi = px.scatter(
            multi_cat, x="n_categorias", y="monto_total",
            size="n_procedimientos", color="tipo proveedor",
            hover_data=["nombre proveedor", "provincia"],
            title="Oferentes por # de categorías vs monto total",
            labels={"n_categorias": "Categorías", "monto_total": "Monto Total (₡)",
                    "n_procedimientos": "Procedimientos"},
        )
        fig_multi.update_layout(height=500)
        st.plotly_chart(fig_multi, use_container_width=True)

        multi_tbl = multi_cat[["nombre proveedor", "cédula proveedor", "n_categorias",
                                "n_procedimientos", "monto_total", "tipo proveedor",
                                "tamaño proveedor"]].copy()
        multi_tbl.columns = ["Proveedor", "Cédula", "Categorías", "Procedimientos",
                              "Monto Total (₡)", "Tipo", "Tamaño"]
        multi_tbl["Monto Total (₡)"] = multi_tbl["Monto Total (₡)"].apply(fmt_crc)
        multi_tbl.index = range(1, len(multi_tbl) + 1)
        st.dataframe(multi_tbl, width="stretch")
    else:
        st.info("No hay oferentes que participen en múltiples categorías con los filtros actuales.")

    st.markdown("---")

    # ── 5. DETALLE OFERENTE ──
    st.subheader("🔍 Detalle por Oferente")

    top_list = prov_agg.head(50)["nombre proveedor"].tolist()
    sel_ofer = st.selectbox("Seleccione un oferente", top_list,
                            index=0 if top_list else None)

    if sel_ofer:
        ofer_data = prov_cat[prov_cat["nombre proveedor"] == sel_ofer].copy()
        ofer_total = ofer_data["monto"].sum()

        do1, do2, do3 = st.columns(3)
        do1.metric("Monto Total", fmt_crc(ofer_total))
        do2.metric("Categorías", f"{ofer_data['categoría'].nunique()}")
        do3.metric("Procedimientos", f"{ofer_data['procedimientos'].sum()}")

        col_ofer_pie, col_ofer_tbl = st.columns([3, 2])

        with col_ofer_pie:
            fig_ofer = px.pie(
                ofer_data, names="categoría", values="monto",
                title=f"Distribución del gasto — {sel_ofer[:50]}",
            )
            st.plotly_chart(fig_ofer, use_container_width=True)

        with col_ofer_tbl:
            ofer_disp = ofer_data[["categoría", "monto", "ordenes", "procedimientos"]].copy()
            ofer_disp["pct"] = (ofer_disp["monto"] / ofer_disp["monto"].sum() * 100).round(1)
            ofer_disp.columns = ["Categoría", "Monto (₡)", "Órdenes", "Procedimientos", "% del Total"]
            ofer_disp = ofer_disp.sort_values("Monto (₡)", ascending=False)
            ofer_disp["Monto (₡)"] = ofer_disp["Monto (₡)"].apply(fmt_crc)
            ofer_disp.index = range(1, len(ofer_disp) + 1)
            st.dataframe(ofer_disp, width="stretch", hide_index=True)

    st.markdown("---")

    # ── Full oferente table ──
    st.subheader("📋 Tabla Completa de Oferentes")
    ofer_full = prov_agg[["nombre proveedor", "cédula proveedor", "monto_total",
                           "n_ordenes", "n_procedimientos", "n_categorias",
                           "tipo proveedor", "tamaño proveedor", "provincia"]].copy()
    ofer_full.columns = ["Proveedor", "Cédula", "Monto Total (₡)", "Órdenes",
                          "Procedimientos", "Categorías", "Tipo", "Tamaño", "Provincia"]
    ofer_full["Monto Total (₡)"] = ofer_full["Monto Total (₡)"].apply(fmt_crc)
    ofer_full.index = range(1, len(ofer_full) + 1)
    st.dataframe(ofer_full, width="stretch", height=600)

    csv_ofer = prov_agg.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar oferentes (CSV)", csv_ofer,
                       "oferentes_completo.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: CORRELACIONES LEGALES
# ══════════════════════════════════════════════════════════════════════════════

with tab_legal:

    st.header("⚖️ Correlaciones Legales de Auditoría")
    st.markdown("""
    Análisis cruzado de **leyes**, **casos judiciales** y **hallazgos de auditoría**
    de la Contraloría General de la República. Las correlaciones se calculan
    automáticamente a partir de leyes citadas y coincidencias institucionales.
    """)

    # ── Load CSV data ──
    @st.cache_data(ttl=3600)
    def _load_legal_data():
        laws = pd.read_csv(DATA_DIR / "sinalevi_sample.csv", sep=";")
        cases = pd.read_csv(DATA_DIR / "nexus_sample.csv", sep=";")
        audit = pd.read_csv(DATA_DIR / "contraloria_sample.csv", sep=";")
        return laws, cases, audit

    @st.cache_data(ttl=3600)
    def _compute_correlations(_laws, _cases, _audit):
        correlations = []
        corr_id = 0

        # 1. Audit → Law (laws cited in audit findings)
        for _, finding in _audit.iterrows():
            cited = str(finding.get("leyes_citadas", ""))
            if not cited or cited == "nan":
                continue
            for _, law in _laws.iterrows():
                law_num = str(law.get("numero_ley", ""))
                if law_num and law_num in cited:
                    corr_id += 1
                    correlations.append({
                        "id": corr_id,
                        "Fuente A": "Hallazgo CGR",
                        "ID A": finding.get("numero_informe", ""),
                        "Título A": str(finding.get("titulo_hallazgo", ""))[:60],
                        "Fuente B": "Ley",
                        "ID B": law_num,
                        "Título B": str(law.get("titulo", ""))[:60],
                        "Tipo": "Cita Legal en Hallazgo",
                        "Confianza": 0.95,
                        "Método": "rule_based",
                        "Evidencia": f"Hallazgo cita {law_num}",
                    })

        # 2. Case → Law (laws cited in judicial cases)
        for _, case in _cases.iterrows():
            cited = str(case.get("leyes_citadas", ""))
            if not cited or cited == "nan":
                continue
            for _, law in _laws.iterrows():
                law_num = str(law.get("numero_ley", ""))
                if law_num and law_num in cited:
                    corr_id += 1
                    correlations.append({
                        "id": corr_id,
                        "Fuente A": "Caso Judicial",
                        "ID A": case.get("numero_expediente", ""),
                        "Título A": str(case.get("materia", ""))[:60],
                        "Fuente B": "Ley",
                        "ID B": law_num,
                        "Título B": str(law.get("titulo", ""))[:60],
                        "Tipo": "Cita Legal en Caso",
                        "Confianza": 0.95,
                        "Método": "rule_based",
                        "Evidencia": f"Caso cita {law_num}",
                    })

        # 3. Audit ↔ Case (same institution)
        for _, finding in _audit.iterrows():
            inst_a = str(finding.get("cedula_institucion", ""))
            if not inst_a or inst_a == "nan":
                continue
            for _, case in _cases.iterrows():
                inst_b = str(case.get("cedula_institucion", ""))
                if inst_a == inst_b:
                    corr_id += 1
                    correlations.append({
                        "id": corr_id,
                        "Fuente A": "Hallazgo CGR",
                        "ID A": finding.get("numero_informe", ""),
                        "Título A": str(finding.get("titulo_hallazgo", ""))[:60],
                        "Fuente B": "Caso Judicial",
                        "ID B": case.get("numero_expediente", ""),
                        "Título B": str(case.get("materia", ""))[:60],
                        "Tipo": "Misma Institución",
                        "Confianza": 0.70,
                        "Método": "rule_based",
                        "Evidencia": f"Institución: {finding.get('nombre_institucion', '')}",
                    })

        return pd.DataFrame(correlations) if correlations else pd.DataFrame()

    laws_df, cases_df, audit_df = _load_legal_data()
    corr_df = _compute_correlations(laws_df, cases_df, audit_df)

    # ── KPIs ──
    kl1, kl2, kl3, kl4 = st.columns(4)
    kl1.metric("📜 Leyes", len(laws_df))
    kl2.metric("⚖️ Casos Judiciales", len(cases_df))
    kl3.metric("🔍 Hallazgos CGR", len(audit_df))
    kl4.metric("🔗 Correlaciones", len(corr_df))

    st.markdown("---")

    sub_laws, sub_cases, sub_audit, sub_corr = st.tabs([
        "📜 Leyes (SINALEVI)", "⚖️ Casos (Nexus)",
        "🔍 Hallazgos (Contraloría)", "🔗 Correlaciones"
    ])

    # ── Sub-tab: Leyes ──
    with sub_laws:
        st.subheader("📜 Marco Legal — SINALEVI")
        laws_disp = laws_df[["numero_ley", "titulo", "tipo", "organo_emisor",
                              "fecha_publicacion", "estado", "materia"]].copy()
        laws_disp.columns = ["Número", "Título", "Tipo", "Órgano Emisor",
                              "Fecha Publicación", "Estado", "Materia"]
        st.dataframe(laws_disp, use_container_width=True, hide_index=True)

        sel_law = st.selectbox(
            "Ver detalle de ley:",
            [f"{r['numero_ley']} — {r['titulo']}" for _, r in laws_df.iterrows()],
            key="legal_law_sel",
        )
        if sel_law:
            law_num = sel_law.split(" — ")[0]
            d = laws_df[laws_df["numero_ley"] == law_num].iloc[0]
            col1, col2 = st.columns(2)
            col1.markdown(f"**Número:** {d.get('numero_ley', '—')}")
            col1.markdown(f"**Tipo:** {d.get('tipo', '—')}")
            col1.markdown(f"**Estado:** {d.get('estado', '—')}")
            col2.markdown(f"**Órgano emisor:** {d.get('organo_emisor', '—')}")
            col2.markdown(f"**Fecha publicación:** {d.get('fecha_publicacion', '—')}")
            col2.markdown(f"**Materia:** {d.get('materia', '—')}")
            if pd.notna(d.get("resumen")):
                st.markdown(f"**Resumen:** {d['resumen']}")
            if pd.notna(d.get("etiquetas")):
                st.markdown(f"**Etiquetas:** {d['etiquetas']}")

    # ── Sub-tab: Casos ──
    with sub_cases:
        st.subheader("⚖️ Casos Judiciales — Nexus")
        cases_disp = cases_df[["numero_expediente", "tribunal", "jurisdiccion",
                                "tipo_caso", "nombre_institucion", "estado",
                                "fecha_interposicion"]].copy()
        cases_disp.columns = ["Expediente", "Tribunal", "Jurisdicción", "Tipo",
                               "Institución", "Estado", "Fecha"]
        st.dataframe(cases_disp, use_container_width=True, hide_index=True)

        sel_case = st.selectbox(
            "Ver detalle de caso:",
            cases_df["numero_expediente"].tolist(),
            key="legal_case_sel",
        )
        if sel_case:
            d = cases_df[cases_df["numero_expediente"] == sel_case].iloc[0]
            col1, col2 = st.columns(2)
            col1.markdown(f"**Expediente:** {d.get('numero_expediente', '—')}")
            col1.markdown(f"**Tribunal:** {d.get('tribunal', '—')}")
            col1.markdown(f"**Jurisdicción:** {d.get('jurisdiccion', '—')}")
            col1.markdown(f"**Estado:** {d.get('estado', '—')}")
            col2.markdown(f"**Demandante:** {d.get('demandante', '—')}")
            col2.markdown(f"**Demandado:** {d.get('demandado', '—')}")
            monto = d.get("monto_disputado", 0)
            col2.markdown(f"**Monto disputado:** ₡{monto:,.0f}" if pd.notna(monto) and monto > 0 else "**Monto:** —")
            col2.markdown(f"**Institución:** {d.get('nombre_institucion', '—')}")
            if pd.notna(d.get("materia")):
                st.markdown(f"**Materia:** {d['materia']}")
            if pd.notna(d.get("resumen_resolucion")):
                st.markdown(f"**Resolución:** {d['resumen_resolucion']}")
            if pd.notna(d.get("leyes_citadas")):
                st.markdown(f"**Leyes citadas:** {d['leyes_citadas']}")

    # ── Sub-tab: Hallazgos ──
    with sub_audit:
        st.subheader("🔍 Hallazgos de Auditoría — Contraloría")
        audit_disp = audit_df[["numero_informe", "titulo_hallazgo", "tipo_auditoria",
                                "area_auditoria", "severidad", "nombre_institucion",
                                "fecha_informe", "estado_disposicion", "monto_observado"]].copy()
        audit_disp.columns = ["Informe", "Hallazgo", "Tipo", "Área", "Severidad",
                               "Institución", "Fecha", "Estado Disposición", "Monto Observado"]
        audit_disp["Monto Observado"] = audit_disp["Monto Observado"].apply(
            lambda v: fmt_crc(v) if pd.notna(v) and v > 0 else "—"
        )

        severities = sorted(audit_df["severidad"].dropna().unique())
        sel_sev = st.multiselect("Filtrar por severidad:", severities,
                                  default=severities, key="legal_sev_filter")
        filtered_audit = audit_disp[audit_disp["Severidad"].isin(sel_sev)]
        st.dataframe(filtered_audit, use_container_width=True, hide_index=True)

        if len(filtered_audit) > 0:
            sev_counts = filtered_audit["Severidad"].value_counts()
            st.bar_chart(sev_counts)

        sel_finding = st.selectbox(
            "Ver detalle:",
            [f"{r['numero_informe']} — {r['titulo_hallazgo']}" for _, r in
             audit_df[audit_df["severidad"].isin(sel_sev)].iterrows()],
            key="legal_audit_sel",
        )
        if sel_finding:
            report_num = sel_finding.split(" — ")[0]
            title = sel_finding.split(" — ", 1)[1]
            match = audit_df[(audit_df["numero_informe"] == report_num) &
                             (audit_df["titulo_hallazgo"] == title)]
            if len(match) > 0:
                d = match.iloc[0]
                col1, col2 = st.columns(2)
                col1.markdown(f"**Informe:** {d.get('numero_informe', '—')}")
                col1.markdown(f"**Severidad:** {d.get('severidad', '—')}")
                col1.markdown(f"**Estado disposición:** {d.get('estado_disposicion', '—')}")
                col2.markdown(f"**Institución:** {d.get('nombre_institucion', '—')}")
                col2.markdown(f"**Área:** {d.get('area_auditoria', '—')}")
                monto = d.get("monto_observado", 0)
                col2.markdown(f"**Monto observado:** ₡{monto:,.0f}" if pd.notna(monto) and monto > 0 else "**Monto:** —")
                if pd.notna(d.get("descripcion_hallazgo")):
                    st.markdown(f"**Descripción:** {d['descripcion_hallazgo']}")
                if pd.notna(d.get("recomendacion")):
                    st.info(f"**Recomendación:** {d['recomendacion']}")
                if pd.notna(d.get("leyes_citadas")):
                    st.markdown(f"**Leyes citadas:** {d['leyes_citadas']}")

    # ── Sub-tab: Correlaciones ──
    with sub_corr:
        st.subheader("🔗 Red de Correlaciones")

        if len(corr_df) > 0:
            # Summary metrics
            type_counts = corr_df["Tipo"].value_counts()
            cols_metric = st.columns(len(type_counts))
            for i, (t, cnt) in enumerate(type_counts.items()):
                cols_metric[i].metric(t, cnt)

            # Filter by type
            sel_corr_type = st.multiselect(
                "Filtrar por tipo:", type_counts.index.tolist(),
                default=type_counts.index.tolist(), key="legal_corr_type",
            )
            filtered_corr = corr_df[corr_df["Tipo"].isin(sel_corr_type)]
            st.dataframe(
                filtered_corr[["Fuente A", "ID A", "Título A",
                                "Fuente B", "ID B", "Título B",
                                "Tipo", "Confianza", "Evidencia"]],
                use_container_width=True, hide_index=True,
            )

            # Visual network
            st.subheader("Mapa de Correlaciones")
            for _, row in filtered_corr.iterrows():
                conf = row["Confianza"]
                emoji = "🟢" if conf >= 0.9 else "🟡" if conf >= 0.6 else "🔴"
                st.markdown(
                    f"{emoji} **{row['Fuente A']}** `{row['Título A']}` "
                    f"↔ **{row['Fuente B']}** `{row['Título B']}` "
                    f"— _{row['Tipo']}_ ({conf:.0%})"
                )

            # Law citation frequency
            st.markdown("---")
            st.subheader("📊 Leyes más citadas")
            law_cites = corr_df[corr_df["Fuente B"] == "Ley"]["ID B"].value_counts()
            if len(law_cites) > 0:
                cite_df = law_cites.reset_index()
                cite_df.columns = ["Ley", "Citas"]
                fig_cite = px.bar(
                    cite_df, x="Ley", y="Citas",
                    title="Frecuencia de Citas por Ley",
                    color="Citas", color_continuous_scale="Reds",
                )
                fig_cite.update_layout(height=400, coloraxis_showscale=False)
                st.plotly_chart(fig_cite, use_container_width=True)
        else:
            st.info("No se encontraron correlaciones.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: GLOSARIO
# ══════════════════════════════════════════════════════════════════════════════

with tab_glosario:

    st.header("📖 Glosario de Tipos de Procedimiento")
    st.markdown("""
    Referencia rápida de los tipos de procedimiento de contratación pública
    que aparecen en los datos de SICOP. Se incluyen tanto figuras de la
    legislación anterior como las establecidas por la **Ley General de
    Contratación Pública (LGCP)**.
    """)

    st.markdown("---")

    glossary_data = pd.DataFrame([
        ("Contratación Directa",
         "Procedimiento utilizado bajo la legislación anterior para adquisiciones de menor cuantía o situaciones específicas autorizadas por la normativa. Generalmente implicaba una competencia más limitada que las licitaciones públicas."),
        ("Licitación Abreviada",
         "Procedimiento de concurso de la legislación anterior aplicado a contrataciones de cuantía intermedia. Requería competencia entre oferentes, pero con requisitos y plazos más simplificados que una licitación pública."),
        ("Licitación Pública Nacional",
         "Procedimiento abierto en el que cualquier proveedor que cumpliera los requisitos podía presentar oferta. Era el mecanismo utilizado para contrataciones de mayor cuantía bajo el régimen anterior."),
        ("Licitación Mayor",
         "Procedimiento ordinario de contratación establecido por la LGCP para las contrataciones de mayor monto o cuantía inestimable. Tiene los requisitos más rigurosos de publicidad, competencia y evaluación."),
        ("Licitación Menor",
         "Procedimiento ordinario de la LGCP aplicable a contrataciones de monto intermedio. Mantiene la competencia entre oferentes pero con plazos y requisitos más ágiles que la Licitación Mayor."),
        ("Licitación Reducida",
         "Procedimiento ordinario de la LGCP para contrataciones de menor cuantía dentro del régimen ordinario. Utiliza plazos más cortos y una tramitación simplificada."),
        ("Procedimiento por Excepción",
         "Contratación que se aparta de los procedimientos ordinarios debido a circunstancias expresamente autorizadas por la ley, como exclusividad, urgencia, acuerdos internacionales u otros supuestos específicos. Su uso debe justificarse y documentarse de forma rigurosa."),
        ("Procedimientos Especiales",
         "Mecanismos de contratación regulados para situaciones particulares que poseen reglas propias distintas de los procedimientos ordinarios. Incluyen figuras como convenios marco, subastas inversas, remates y otros esquemas previstos por la LGCP."),
    ], columns=["Tipo", "Definición"])

    st.dataframe(glossary_data, hide_index=True, use_container_width=True,
                 column_config={
                     "Tipo": st.column_config.TextColumn(width="medium"),
                     "Definición": st.column_config.TextColumn(width="large"),
                 })

    st.markdown("---")

    st.header("💰 Umbrales de Montos por Tipo de Procedimiento")
    st.markdown("""
    Montos máximos (en colones) que definen el tipo de procedimiento ordinario
    aplicable según el año, régimen y tipo de objeto contractual.
    """)

    umbrales_data = pd.DataFrame([
        (2025, "Ordinario", "Bienes y Servicios", 64_804_338, 233_449_258, 233_449_258),
        (2025, "Ordinario", "Obras", 174_473_216, 697_892_648, 697_892_648),
        (2026, "Ordinario", "Bienes y Servicios", 64_559_795, 258_239_178, 258_239_178),
    ], columns=["Año", "Régimen", "Tipo Objeto", "Licitación Reducida Hasta (₡)",
                "Licitación Menor Hasta (₡)", "Licitación Mayor Desde (₡)"])

    st.dataframe(umbrales_data, hide_index=True, use_container_width=True,
                 column_config={
                     "Año": st.column_config.NumberColumn(format="%d"),
                     "Licitación Reducida Hasta (₡)": st.column_config.NumberColumn(format="₡%,.0f"),
                     "Licitación Menor Hasta (₡)": st.column_config.NumberColumn(format="₡%,.0f"),
                     "Licitación Mayor Desde (₡)": st.column_config.NumberColumn(format="₡%,.0f"),
                 })
