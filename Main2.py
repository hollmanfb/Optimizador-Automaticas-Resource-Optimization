import streamlit as st
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# 1. BASE DE DATOS MAESTRA (MATRIZ DE M&T MEDMIX)
# -------------------------------------------------------------------------
MAX_SATURACION_ESTANDAR = 0.97  
DISTANCIA_CRITICA_MAX = 20.0  

WORKLOAD_MAESTRO = {
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

HEURISTICA_PASILLO = {"922", "911", "926", "925", "905"}
LISTA_7_OPERARIOS = [f"Operario {i}" for i in range(1, 8)]

# -------------------------------------------------------------------------
# 2. ALGORITMO SEGURO DE BALANCEO DE LÍNEA (IA SIN DUPLICADOS)
# -------------------------------------------------------------------------
def optimizar_con_operarios_fijos(maquinas_trabajando, operarios_disponibles):
    asignacion = {op: [] for op in LISTA_7_OPERARIOS}
    maquinas_por_asignar = [m for m in maquinas_trabajando]

    if not operarios_disponibles:
        return asignacion

    # Fase 1: Celdas Dedicadas Estandarizadas (Carga >= 100%)
    ops_pool = [o for o in operarios_disponibles]
    for m in list(maquinas_por_asignar):
        if WORKLOAD_MAESTRO.get(m, 0) >= 1.00:
            if ops_pool:
                op_elegido = ops_pool.pop(0)
                asignacion[op_elegido].append(m)
                maquinas_por_asignar.remove(m)

    # Ordenar por criticidad de carga para balanceo óptimo
    maquinas_por_asignar.sort(key=lambda x: -WORKLOAD_MAESTRO.get(x, 0))

    # Fase 2: Distribución por Distancia y Confort (< 97%)
    for m in list(maquinas_por_asignar):
        mejor_op = None
        menor_distancia_op = float('inf')
        
        for op in ops_pool:  # Solo usar operarios que no tengan celdas dedicadas de 100%
            maqs_del_op = asignacion[op]
            carga_actual = sum([WORKLOAD_MAESTRO[x] for x in maqs_del_op])
            
            todas_en_pasillo = all(x in HEURISTICA_PASILLO for x in maqs_del_op + [m])
            tope_limite = 1.30 if todas_en_pasillo else MAX_SATURACION_ESTANDAR
            
            if carga_actual + WORKLOAD_MAESTRO[m] <= tope_limite:
                if any(MATRIZ_DISTANCIAS.get(m, {}).get(ya, 0) > DISTANCIA_CRITICA_MAX for ya in maqs_del_op):
                    continue
                
                dist_eval = 0.0 if not maqs_del_op else np.mean([MATRIZ_DISTANCIAS[m].get(ya, 50.0) for ya in maqs_del_op])
                if dist_eval < menor_distancia_op:
                    menor_distancia_op = dist_eval
                    mejor_op = op
                    
        if mejor_op:
            asignacion[mejor_op].append(m)
            maquinas_por_asignar.remove(m)

    # Fase 3: Desborde Técnico Controlado (Si faltan operarios, meter por cercanía física estricta)
    for m in list(maquinas_por_asignar):
        mejor_op_desborde = None
        menor_distancia_desborde = float('inf')
        
        for op in operarios_disponibles:
            maqs_del_op = asignacion[op]
            
            # Bloqueo estricto de distancias largas para evitar traslados inviables en planta
            if any(MATRIZ_DISTANCIAS.get(m, {}).get(ya, 0) > DISTANCIA_CRITICA_MAX for ya in maqs_del_op):
                continue
                
            dist_eval = 0.0 if not maqs_del_op else np.mean([MATRIZ_DISTANCIAS[m].get(ya, 50.0) for ya in maqs_del_op])
            if dist_eval < menor_distancia_desborde:
                menor_distancia_desborde = dist_eval
                mejor_op_desborde = op
                
        if mejor_op_desborde:
            asignacion[mejor_op_desborde].append(m)
            maquinas_por_asignar.remove(m)

    return asignacion

# -------------------------------------------------------------------------
# 3. INTERFAZ GRÁFICA CONFIGURABLE
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador de Turnos medmix")

if "estados_maquinas" not in st.session_state:
    st.session_state.estados_maquinas = {m: "Trabajando" for m in WORKLOAD_MAESTRO.keys()}
    # Escenario inicial: algunas celdas en reposo semanal
    for desactiva in ["904", "906", "916", "917", "925", "926", "928"]:
        st.session_state.estados_maquinas[desactiva] = "Día Libre"

if "estados_operarios" not in st.session_state:
    st.session_state.estados_operarios = {op: "Disponible" if idx < 4 else "Día Libre / Ausente" for idx, op in enumerate(LISTA_7_OPERARIOS)}

# --- PANEL DE MANDOS LATERAL ---
with st.sidebar:
    st.image("https://www.medmix.mixpac.com/images/medmix_Logo_Pos_RGB.svg", width=180)
    
    st.markdown("### 👤 Estado del Personal")
    cambio_operarios = False
    estados_ops_actualizados = {}
    for op in LISTA_7_OPERARIOS:
        estado_previo_op = st.session_state.estados_operarios.get(op, "Disponible")
        sel_op = st.selectbox(f"{op}:", options=["Disponible", "Día Libre / Ausente"], index=0 if estado_previo_op == "Disponible" else 1, key=f"sel_status_{op}")
        estados_ops_actualizados[op] = sel_op
        if sel_op != estado_previo_op:
            cambio_operarios = True
            
    if cambio_operarios:
        st.session_state.estados_operarios = estados_ops_actualizados

    st.markdown("---")
    st.markdown("### ⚙️ Celdas en Operación")
    
    cambio_maquinas = False
    for m in sorted(WORKLOAD_MAESTRO.keys()):
        estado_actual = st.session_state.estados_maquinas.get(m, "Trabajando")
        st.markdown(f"**Celda {m}** (Carga: {WORKLOAD_MAESTRO[m]*100:.1f}%)")
        
        c_tr, c_dl = st.columns(2)
        with c_tr:
            if st.button("🟢 Activa" if estado_actual == "Trabajando" else "Activa", key=f"btn_tr_{m}", use_container_width=True):
                st.session_state.estados_maquinas[m] = "Trabajando"; cambio_maquinas = True
        with c_dl:
            if st.button("🔴 Parada" if estado_actual == "Día Libre" else "Parada", key=f"btn_dl_{m}", use_container_width=True):
                st.session_state.estados_maquinas[m] = "Día Libre"; cambio_maquinas = True

# Ejecución única y centralizada del algoritmo (Evita descalces visuales)
maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]
propuesta_turno = optimizar_con_operarios_fijos(maquinas_activas, ops_activos)

# -------------------------------------------------------------------------
# 4. DASHBOARD CENTRAL Y ALERTAS DE CAPACIDAD
# -------------------------------------------------------------------------
st.title("🏭 Panel de Balanceo de Celdas — Área de Montaje")

# Validación física de cobertura
todas_las_maquinas_asignadas = []
for list_m in propuesta_turno.values():
    todas_las_maquinas_asignadas.extend(list_m)
maquinas_huerfanas = set(maquinas_activas) - set(todas_las_maquinas_asignadas)

if maquinas_huerfanas:
    st.error(f"🚨 **MENSJAE DE ALERTA: FALTAN MÁQUINAS POR ASIGNAR.** Las celdas {', '.join(sorted(maquinas_huerfanas))} están en marcha pero no hay personal suficiente a la distancia requerida. Por favor, pasa un nuevo operario a 'Disponible' en el panel izquierdo.")
else:
    st.success("✅ Estructura Estable: El 100% de las celdas activas cuentan con cobertura.")

st.markdown("---")

# -------------------------------------------------------------------------
# 5. DESPLIEGUE SEGURO DE FICHAS DE TRABAJO (MATRIZ REAL DE PLANTA)
# -------------------------------------------------------------------------
st.subheader("📋 Distribución Real del Turno")
cols_tarjetas = st.columns(4)

for idx, operario in enumerate(LISTA_7_OPERARIOS):
    esta_disponible = st.session_state.estados_operarios.get(operario, "Disponible") == "Disponible"
    maquinas_del_op = propuesta_turno.get(operario, [])
    
    with cols_tarjetas[idx % 4]:
        if not esta_disponible:
            with st.container(border=True):
                st.markdown(f"<h3 style='color: #b0b0b0;'>👤 {operario}</h3>", unsafe_allow_html=True)
                st.caption("⚠️ **Ausente / Descanso**")
        else:
            with st.container(border=True):
                st.markdown(f"### 👤 {operario}")
                
                if not maquinas_del_op:
                    st.info("💤 **Puesto de Reserva / Apoyo**")
                    st.caption("Carga de Trabajo: 0.0%")
                else:
                    # Mostrar las máquinas asignadas de manera limpia y clara
                    st.markdown("**Celdas a Cargo:**")
                    for m in maquinas_del_op:
                        st.write(f"• **Máquina {m}** ({WORKLOAD_MAESTRO[m]*100:.1f}% carga)")
                    
                    # Cálculo exacto de carga real asignada
                    carga_total_op = sum([WORKLOAD_MAESTRO[m] for m in maquinas_del_op]) * 100
                    
                    # Alertas de saturación exactas por ficha
                    if carga_total_op > 100.0:
                        st.error(f"🔴 Sobrecarga Crítica: {carga_total_op:.1f}%")
                    elif carga_total_op > 97.0:
                        st.warning(f"⚠️ Carga Elevada: {carga_total_op:.1f}%")
                    else:
                        st.success(f"⚡ Carga Óptima: {carga_total_op:.1f}%")
                    
                    # Despliegue de Trayectos Internos de la Ficha
                    if len(maquinas_del_op) > 1:
                        with st.expander("📍 Auditoría de Distancias"):
                            for i in range(len(maquinas_del_op)):
                                for j in range(i + 1, len(maquinas_del_op)):
                                    m1, m2 = maquinas_del_op[i], maquinas_del_op[j]
                                    d = MATRIZ_DISTANCIAS.get(m1, {}).get(m2, 0)
                                    st.write(f"{m1} ↔️ {m2}: {d} metros")
