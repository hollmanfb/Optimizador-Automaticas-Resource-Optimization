import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# -------------------------------------------------------------------------
# 1. BASE DE DATOS MAESTRA (MÉTODOS Y TIEMPOS - MEDMIX)
# -------------------------------------------------------------------------
MAX_SATURACION_ESTANDAR = 1.10  
DISTANCIA_CRITICA_MAX = 20.0  

WORKLOAD_MAESTRO_BASE = {
    "902": 0.3712, "903": 0.3437, "904": 0.3016, "905": 0.3218,
    "906": 0.3289, "907": 0.3217, "911": 0.1821, "916": 0.3868,
    "917": 1.0000, "922": 0.5321, "923": 0.6995, "924": 1.0000,
    "925": 0.3356, "926": 0.3361, "927": 0.5300, "928": 0.6735
}

MATRIZ_DISTANCIAS = {
    "902": {"902":0, "903":7, "904":23, "905":9, "906":4, "907":8, "911":25, "916":21, "917":41, "922":21, "923":27, "924":15, "925":12, "926":26, "927":6, "928":40},
    "903": {"902":7, "903":0, "904":27, "905":10, "906":8, "907":8.5, "911":20, "916":26, "917":48, "922":21, "923":26, "924":12, "925":15, "926":20, "927":11, "928":47},
    "904": {"902":23, "903":27, "904":0, "905":39, "906":26, "907":29, "911":44, "916":3, "917":27, "922":43, "923":13, "924":24, "925":42, "926":45, "927":15, "928":21},
    "905": {"902":9, "903":10, "904":39, "905":0, "906":12, "907":10, "911":9, "916":28, "917":50, "922":12, "923":28, "924":18, "925":3, "926":6, "927":18, "928":35},
    "906": {"902":4, "903":8, "904":26, "905":12, "906":0, "907":2, "911":20, "916":23, "917":49, "922":19, "923":36, "924":20, "925":18, "926":21, "927":12, "928":37},
    "907": {"902":8, "903":8.5, "904":29, "905":10, "906":2, "907":0, "911":18, "916":25, "917":51, "922":17, "923":38, "924":22, "925":16, "926":19, "927":14, "928":39},
    "911": {"902":25, "903":20, "904":44, "905":9, "906":20, "907":18, "911":0, "916":37, "917":59, "922":1, "923":37, "924":27, "925":3, "926":1, "927":27, "928":44},
    "916": {"902":21, "903":26, "904":3, "905":28, "906":23, "907":25, "911":37, "916":0, "917":27, "922":40, "923":10, "924":21, "925":39, "926":42, "927":12, "928":26},
    "917": {"902":41, "903":48, "904":23, "905":50, "906":49, "907":51, "911":59, "916":27, "917":0, "922":70, "923":38, "924":74, "925":78, "926":80, "927":47, "928":16},
    "922": {"902":21, "903":21, "904":43, "905":12, "906":19, "907":17, "911":1, "916":40, "917":70, "922":0, "923":50, "924":36, "925":10, "926":13, "927":37, "928":62},
    "923": {"902":27, "903":26, "904":13, "905":28, "906":36, "907":38, "911":37, "916":10, "917":38, "922":50, "923":0, "924":19, "925":42, "926":46, "927":22, "928":46},
    "924": {"902":15, "903":12, "904":24, "905":18, "906":20, "907":22, "911":27, "916":21, "917":74, "922":36, "923":19, "924":0, "925":24, "926":28, "927":20, "928":45},
    "925": {"902":12, "903":15, "904":42, "905":3, "906":18, "907":16, "911":3, "916":39, "917":78, "922":10, "923":42, "924":24, "925":0, "926":3, "927":35, "928":62},
    "926": {"902":26, "903":20, "904":45, "905":6, "906":21, "907":19, "911":1, "916":42, "917":80, "922":13, "923":46, "924":28, "925":3, "926":0, "927":38, "928":66},
    "927": {"902":6, "903":11, "904":15, "905":18, "906":12, "907":14, "911":27, "916":12, "917":47, "922":37, "923":22, "924":20, "925":35, "926":38, "927":0, "928":25},
    "928": {"902":40, "903":40, "904":21, "905":35, "906":37, "907":39, "911":44, "916":26, "917":16, "922":62, "923":46, "924":45, "925":62, "926":66, "927":25, "928":0}
}

LISTA_8_OPERARIOS = [f"Operario {i}" for i in range(1, 9)]

# -------------------------------------------------------------------------
# 2. MOTOR IA CON RESTRICCIONES TÉCNICAS REFORZADAS
# -------------------------------------------------------------------------
def optimizar_con_operarios_fijos(maquinas_trabajando, operarios_disponibles, cargas_activas, variante_902):
    asignacion = {op: [] for op in LISTA_8_OPERARIOS}
    maquinas_por_asignar = [m for m in maquinas_trabajando]

    if not operarios_disponibles:
        return asignacion

    # Regla 1: Asignaciones de celdas Hito Estructurales (100% de carga)
    mapeo_estricto = {"917": "Operario 4", "924": "Operario 6", "928": "Operario 7"}
    for m, op in mapeo_estricto.items():
        if m in maquinas_por_asignar and op in operarios_disponibles:
            asignacion[op].append(m)
            maquinas_por_asignar.remove(m)

    # Regla 2: Gestión de la Celda 902 y 927 (Control de Cánula Larga)
    if "Operario 1" in operarios_disponibles:
        if "927" in maquinas_por_asignar:
            asignacion["Operario 1"].append("927")
            maquinas_por_asignar.remove("927")
        
        # SÓLO se permite emparejar con 902 si NO es Cánula Larga
        if variante_902 == "Cánula Corta (37.1%)" and "902" in maquinas_por_asignar:
            asignacion["Operario 1"].append("902")
            maquinas_por_asignar.remove("902")

    # Regla 3: Bloque del Pasillo Operario 2 (922, 911, 905)
    if "Operario 2" in operarios_disponibles:
        for m in ["922", "911", "905"]:
            if m in maquinas_por_asignar:
                asignacion["Operario 2"].append(m)
                maquinas_por_asignar.remove(m)

    # Regla 4: Bloque Operario 3 (906, 907, 903)
    if "Operario 3" in operarios_disponibles:
        for m in ["906", "907", "903"]:
            if m in maquinas_por_asignar:
                asignacion["Operario 3"].append(m)
                maquinas_por_asignar.remove(m)

    # Ordenar las celdas huérfanas de mayor a menor criticidad de carga para compactar
    maquinas_por_asignar.sort(key=lambda x: -cargas_activas.get(x, 0))

    # Filtrar operarios que realmente tienen espacio disponible para evitar subutilización
    for m in list(maquinas_por_asignar):
        # Si es la 902 en Cánula Larga, vetamos explícitamente al Operario 1
        ops_elegibles = [
            o for o in operarios_disponibles 
            if not (m == "902" and variante_902 == "Cánula Larga (45.0%)" and o == "Operario 1")
        ]
        
        mejor_op = None
        menor_distancia = float('inf')
        
        # Buscar operario óptimo por cercanía física que no supere el límite máximo
        for op in ops_elegibles:
            maqs_del_op = asignacion[op]
            carga_actual = sum([cargas_activas[x] for x in maqs_del_op])
            
            if carga_actual + cargas_activas[m] <= MAX_SATURACION_ESTANDAR:
                if maqs_del_op:
                    dist_eval = np.mean([MATRIZ_DISTANCIAS[m].get(ya, 50.0) for ya in maqs_del_op])
                else:
                    dist_eval = 0.0
                
                if dist_eval < menor_distancia:
                    menor_distancia = dist_eval
                    mejor_op = op
                    
        if mejor_op:
            asignacion[mejor_op].append(m)
            maquinas_por_asignar.remove(m)

    # Forzar asignación de celdas restantes al operario con menor carga actual (Garantía anti-huecos)
    for m in list(maquinas_por_asignar):
        ops_posibles = [
            o for o in operarios_disponibles 
            if not (m == "902" and variante_902 == "Cánula Larga (45.0%)" and o == "Operario 1")
        ]
        if ops_posibles:
            op_menos_cargado = min(ops_posibles, key=lambda o: sum([cargas_activas[x] for x in asignacion[o]]))
            asignacion[op_menos_cargado].append(m)
            maquinas_por_asignar.remove(m)

    return asignacion

# -------------------------------------------------------------------------
# 3. CONFIGURACIÓN DE INTERFAZ Y SESIÓN
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador de Turnos medmix")

if "version_902" not in st.session_state:
    st.session_state.version_902 = "Cánula Corta (37.1%)"

cargas_dinamicas_turno = WORKLOAD_MAESTRO_BASE.copy()
cargas_dinamicas_turno["902"] = 0.4500 if st.session_state.version_902 == "Cánula Larga (45.0%)" else 0.3712

st.markdown("""
    <style>
    @media print {
        header, [data-testid="stSidebar"], .stButton, button, footer, hr, 
        [data-testid="stMetricWidget"], .stAlert, [data-baseinput="true"], .stMultiSelect {
            display: none !important;
        }
        [data-testid="stMainBlockContainer"] { padding: 0px !important; margin: 0px !important; max-width: 100% !important; }
        div[data-testid="stBlock"] div[data-testid="stBlock"] { display: none !important; }
        .print-only-card {
            border: 1px solid #000 !important; padding: 6px !important;
            margin-bottom: 4px !important; border-radius: 4px !important;
            background-color: #fff !important; page-break-inside: avoid !important;
        }
        .print-header-title { text-align: center !important; font-size: 18px !important; margin-bottom: 10px !important; }
    }
    </style>
""", unsafe_allow_html=True)

if "estados_maquinas" not in st.session_state:
    st.session_state.estados_maquinas = {m: "Trabajando" for m in cargas_dinamicas_turno.keys()}
    for desactiva in ["904", "916", "925", "926"]:
        st.session_state.estados_maquinas[desactiva] = "Día Libre"

if "estados_operarios" not in st.session_state:
    st.session_state.estados_operarios = {op: "Disponible" for op in LISTA_8_OPERARIOS}

if "prioridades_estrellas" not in st.session_state:
    st.session_state.prioridades_estrellas = {m: "⭐⭐ Media" for m in cargas_dinamicas_turno.keys()}

maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]

if "propuesta_actual" not in st.session_state:
    st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos, cargas_dinamicas_turno, st.session_state.version_902)

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.image("https://www.medmix.mixpac.com/images/medmix_Logo_Pos_RGB.svg", width=180)
    
    st.markdown("### 👤 Asistencia del Turno")
    for op in LISTA_8_OPERARIOS:
        estado_previo = st.session_state.estados_operarios.get(op, "Disponible")
        sel_op = st.selectbox(f"{op}:", options=["Disponible", "Día Libre / Ausente"], index=0 if estado_previo == "Disponible" else 1, key=f"sel_status_{op}")
        if sel_op != estado_previo:
            st.session_state.estados_operarios[op] = sel_op
            ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]
            st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos, cargas_dinamicas_turno, st.session_state.version_902)
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Celdas Activas y Mezcla de Productos")
    for m in sorted(cargas_dinamicas_turno.keys()):
        estado_actual = st.session_state.estados_maquinas.get(m, "Trabajando")
        
        if m == "902":
            st.markdown(f"**Celda 902 — Control de Variante**")
            version_sel = st.selectbox("Configuración de Cánula:", options=["Cánula Corta (37.1%)", "Cánula Larga (45.0%)"], index=0 if st.session_state.version_902 == "Cánula Corta (37.1%)" else 1, key="sel_v902")
            if version_sel != st.session_state.version_902:
                st.session_state.version_902 = version_sel
                cargas_dinamicas_turno["902"] = 0.4500 if version_sel == "Cánula Larga (45.0%)" else 0.3712
                st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos, cargas_dinamicas_turno, st.session_state.version_902)
                st.rerun()
        else:
            st.markdown(f"**Celda {m}** — Carga: {cargas_dinamicas_turno[m]*100:.1f}%")
        
        c_tr, c_dl = st.columns(2)
        with c_tr:
            if st.button("🟢 Activa", key=f"btn_tr_{m}", use_container_width=True, type="primary" if estado_actual == "Trabajando" else "secondary", disabled=(estado_actual == "Trabajando")):
                st.session_state.estados_maquinas[m] = "Trabajando"
                maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
                st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos, cargas_dinamicas_turno, st.session_state.version_902)
                st.rerun()
        with c_dl:
            if st.button("🔴 Parada", key=f"btn_dl_{m}", use_container_width=True, type="primary" if estado_actual == "Día Libre" else "secondary", disabled=(estado_actual == "Día Libre")):
                st.session_state.estados_maquinas[m] = "Día Libre"
                for op in LISTA_8_OPERARIOS:
                    if m in st.session_state.propuesta_actual.get(op, []):
                        st.session_state.propuesta_actual[op].remove(m)
                maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
                st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos, cargas_dinamicas_turno, st.session_state.version_902)
                st.rerun()

# -------------------------------------------------------------------------
# 4. DASHBOARD GENERAL DE RENDIMIENTO
# -------------------------------------------------------------------------
st.markdown("<h2 class='print-header-title'>🏭 Panel de Distribución Optimizada de Mano de Obra</h2>", unsafe_allow_html=True)

num_maquinas_trabajando = len(maquinas_activas)
num_operarios_disponibles = len(ops_activos)

cargas_reales_operarios = []
for op in ops_activos:
    maqs_del_op = st.session_state.propuesta_actual.get(op, [])
    cargas_reales_operarios.append(sum([cargas_dinamicas_turno.get(x, 0) for x in maqs_del_op]))
saturacion_media_turno = (np.mean(cargas_reales_operarios) * 100) if cargas_reales_operarios else 0.0

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1: st.metric(label="📊 Celdas Activas en Producción", value=f"{num_maquinas_trabajando} Máquinas")
with kpi2: st.metric(label="👤 Personal Disponible", value=f"{num_operarios_disponibles} de 8 Ops")
with kpi3: st.metric(label="⚡ Balanceo de Carga General", value=f"{saturacion_media_turno:.1f}%")

todas_las_maquinas_en_uso = []
for op_k in ops_activos:
    todas_las_maquinas_en_uso.extend(st.session_state.propuesta_actual.get(op_k, []))
maquinas_faltantes = set(maquinas_activas) - set(todas_las_maquinas_en_uso)

if maquinas_faltantes:
    st.error(f"⚠️ **ALERTA EXTRAORDINARIA:** Alertas de desborde sin cubrir: {', '.join(sorted(maquinas_faltantes))}")
else:
    st.success("✅ Eficiencia de Cobertura Completa: Ninguna celda activa se encuentra desatendida.")

st.markdown("---")

# -------------------------------------------------------------------------
# 5. MATRIZ DE ASIGNACIÓN EFECTIVA (COMPACTA Y SEGURA)
# -------------------------------------------------------------------------
st.subheader("🚀 Fichas de Operación por Puesto")
cols_res = st.columns(4)

for idx, operario in enumerate(LISTA_8_OPERARIOS):
    esta_disponible = st.session_state.estados_operarios.get(operario, "Disponible") == "Disponible"
    maquinas_del_operario = st.session_state.propuesta_actual.get(operario, [])
    
    with cols_res[idx % 4]:
        st.markdown(f"<div class='print-only-card'>", unsafe_allow_html=True)
        
        if not esta_disponible:
            with st.container(border=True):
                st.markdown(f"<h3 style='color: #999; margin:0;'>👤 {operario}</h3>", unsafe_allow_html=True)
                st.caption("💤 **Ausente / Descanso**")
        else:
            with st.container(border=True):
                st.markdown(f"<h3 style='margin:0;'>👤 {operario}</h3>", unsafe_allow_html=True)
                
                # Bloqueo estricto cruzado
                maquinas_ocupadas_por_otros = []
                for op_ref, maqs_ref in st.session_state.propuesta_actual.items():
                    if op_ref != operario and st.session_state.estados_operarios.get(op_ref) == "Disponible":
                        maquinas_ocupadas_por_otros.extend(maqs_ref)
                
                opciones_libres = sorted(list(set(maquinas_activas) - set(maquinas_ocupadas_por_otros)))
                opciones_visibles = sorted(list(set(opciones_libres) | set(maquinas_del_operario)))

                # RESTRICCIÓN MANUAL: Si es Operario 1 y es Cánula Larga, le impedimos elegir la 902
                if operario == "Operario 1" and st.session_state.version_902 == "Cánula Larga (45.0%)":
                    if "902" in opciones_visibles:
                        opciones_visibles.remove("902")

                nuevas_maquinas = st.multiselect(
                    "Asignación de Celdas:", 
                    options=opciones_visibles, 
                    default=maquinas_del_operario, 
                    key=f"ms_sync_op_{operario}_{hash(tuple(maquinas_del_operario))}"
                )
                
                if nuevas_maquinas != maquinas_del_operario:
                    st.session_state.propuesta_actual[operario] = nuevas_maquinas
                    st.rerun()
                
                if nuevas_maquinas:
                    labels = []
                    for m in nuevas_maquinas:
                        if m == "902":
                            if st.session_state.version_902 == "Cánula Larga (45.0%)":
                                labels.append(f"<span style='background-color:#ffebee; border:1px solid #d32f2f; padding:2px 6px; border-radius:3px; font-weight:bold; color:#d32f2f; margin-right:4px;'>M-902 [C. LARGA]</span>")
                            else:
                                labels.append(f"<span style='background-color:#e8f5e9; border:1px solid #2e7d32; padding:2px 6px; border-radius:3px; font-weight:bold; color:#2e7d32; margin-right:4px;'>M-902 [C. CORTA]</span>")
                        else:
                            labels.append(f"<span style='background-color:#e1f5fe; border:1px solid #0288d1; padding:2px 6px; border-radius:3px; font-weight:bold; margin-right:4px;'>M-{m}</span>")
                    st.markdown(f"<div style='margin-top:8px; margin-bottom:8px;'>{' '.join(labels)}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#cc0000; font-style:italic; font-weight:bold;'>⚠️ Puesto libre sin máquinas asignadas</div>", unsafe_allow_html=True)

                carga_real = sum([cargas_dinamicas_turno.get(m, 0) for m in nuevas_maquinas])
                sat_p = carga_real * 100
                
                if sat_p > 110.0:
                    st.error(f"💥 Sobrecarga: {sat_p:.1f}%")
                elif sat_p > 95.0:
                    st.warning(f"⚠️ Carga Alta: {sat_p:.1f}%")
                elif sat_p == 0:
                    pass
                else:
                    st.success(f"⚡ Carga Óptima: {sat_p:.1f}%")

                # Alerta visual estricta en la tarjeta del Operario 1 si tiene la combinación prohibida por error manual
                if operario == "Operario 1" and "902" in nuevas_maquinas and "927" in nuevas_maquinas and st.session_state.version_902 == "Cánula Larga (45.0%)":
                    st.error("🚨 CRÍTICO: Prohibido juntar 927 + 902 en Cánula Larga.")

                if len(nuevas_maquinas) > 1:
                    distancias_texto = []
                    for i in range(len(nuevas_maquinas)):
                        for j in range(i + 1, len(nuevas_maquinas)):
                            m1, m2 = nuevas_maquinas[i], nuevas_maquinas[j]
                            dist = MATRIZ_DISTANCIAS.get(m1, {}).get(m2, 0)
                            distancias_texto.append(f"{m1} ↔️ {m2}: {dist}m")
                                
                    with st.expander("📍 Trayectos de Celda", expanded=False):
                        for txt in distancias_texto: st.write(txt)
        
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 6. PANEL DE ACCIONES DEFINITIVAS
# -------------------------------------------------------------------------
st.write("---")
c_recalc, c_print = st.columns(2)

with c_recalc:
    if st.button("🔄 Recalcular Distribución por Proximidad Física Real (IA)", type="primary", use_container_width=True):
        st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos, cargas_dinamicas_turno, st.session_state.version_902)
        st.rerun()

with c_print:
    components.html("""
        <button onclick="window.parent.print()" style="
            width: 100%; 
            background-color: #2e7d32; 
            color: white; 
            padding: 10px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            font-family: sans-serif;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.2);">
            🖨️ Imprimir Fichas del Turno (Filtro 1 Hoja Limpia)
        </button>
    """, height=50)
