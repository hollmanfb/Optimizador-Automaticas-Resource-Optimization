import streamlit as st
import numpy as np

# -------------------------------------------------------------------------
# 1. BASE DE DATOS MAESTRA (MÉTODOS Y TIEMPOS - MEDMIX)
# -------------------------------------------------------------------------
# Se actualiza el límite máximo de saturación al 110% según lo solicitado
MAX_SATURACION_ESTANDAR = 1.10  
DISTANCIA_CRITICA_MAX = 20.0  

WORKLOAD_MAESTRO = {
    "902": 0.3712, "903": 0.3437, "904": 0.3016, "905": 0.3218,
    "906": 0.3289, "907": 0.3217, "911": 0.1821, "916": 0.3868,
    "917": 1.0000, "922": 0.5321, "923": 0.6995, "924": 1.0000,
    "925": 0.3356, "926": 0.3361, "927": 0.5300, "928": 0.6735
}

MATRIZ_DISTANCIAS = {
    "902": {"902":0, "903":7, "904":23, "905":9, "906":4, "907":8, "911":25, "916":21, "917":41, "922":21, "923":27, "924":15, "925":12, "926":26, "927":3, "928":40},
    "903": {"902":7, "903":0, "904":27, "905":10, "906":8, "907":8.5, "911":20, "916":26, "917":48, "922":21, "923":26, "924":12, "925":15, "926":20, "927":11, "928":47},
    "904": {"902":23, "903":27, "904":0, "905":39, "906":26, "907":29, "911":44, "916":3, "917":27, "922":43, "923":13, "924":24, "925":42, "926":45, "927":15, "928":21},
    "905": {"902":9, "903":10, "904":39, "905":0, "906":12, "907":10, "911":9, "916":28, "917":50, "922":12, "923":28, "924":18, "925":3, "926":6, "927":18, "928":35},
    "906": {"902":4, "903":8, "904":26, "905":12, "906":0, "907":2, "911":20, "916":23, "917":49, "922":19, "923":36, "924":20, "925":18, "926":21, "927":12, "928":37},
    "907": {"902":8, "903":8.5, "904":29, "905":10, "906":2, "907":0, "911":18, "916":25, "917":51, "922":17, "923":38, "924":22, "925":16, "926":19, "927":18, "928":39},
    "911": {"902":25, "903":20, "904":44, "905":9, "906":20, "907":18, "911":0, "916":37, "917":59, "922":1, "923":37, "924":27, "925":3, "926":1, "927":27, "928":44},
    "916": {"902":21, "903":26, "904":3, "905":28, "906":23, "907":25, "911":37, "916":0, "917":27, "922":40, "923":10, "924":21, "925":39, "926":42, "927":12, "928":26},
    "917": {"902":41, "903":48, "904":23, "905":50, "906":49, "907":51, "911":59, "916":27, "917":0, "922":70, "923":38, "924":74, "925":78, "926":80, "927":47, "928":16},
    "922": {"902":21, "903":21, "904":43, "905":12, "906":19, "907":17, "911":1, "916":40, "917":70, "922":0, "923":50, "924":36, "925":10, "926":13, "927":37, "928":62},
    "923": {"902":27, "903":26, "904":13, "905":28, "906":36, "907":38, "911":37, "916":10, "917":38, "922":50, "923":0, "924":19, "925":42, "926":46, "927":22, "928":46},
    "924": {"902":15, "903":12, "904":24, "905":18, "906":20, "907":22, "911":27, "916":21, "917":74, "922":36, "923":19, "924":0, "925":24, "926":28, "927":20, "928":45},
    "925": {"902":12, "903":15, "904":42, "905":3, "906":18, "907":16, "911":3, "916":39, "917":78, "922":10, "923":42, "924":24, "925":0, "926":3, "927":35, "928":62},
    "926": {"902":26, "903":20, "904":45, "905":6, "906":21, "907":19, "911":1, "916":42, "917":80, "922":13, "923":46, "924":28, "925":3, "926":0, "927":38, "928":66},
    "927": {"902":3, "903":11, "904":15, "905":18, "906":12, "907":18, "911":27, "916":12, "917":47, "922":37, "923":22, "924":20, "925":35, "926":38, "927":0, "928":25},
    "928": {"902":40, "903":40, "904":21, "905":35, "906":37, "907":39, "911":44, "916":26, "917":16, "922":62, "923":46, "924":45, "925":62, "926":66, "927":25, "928":0}
}

HEURISTICA_PASILLO = {"922", "911", "926", "925", "905"}
LISTA_7_OPERARIOS = [f"Operario {i}" for i in range(1, 8)]

# -------------------------------------------------------------------------
# 2. ALGORITMO OPTIMIZADOR ADAPTADO A LAS REFERENCIAS
# -------------------------------------------------------------------------
def optimizar_con_operarios_fijos(maquinas_trabajando, operarios_disponibles):
    asignacion = {op: [] for op in LISTA_7_OPERARIOS}
    maquinas_por_assignar = [m for m in maquinas_trabajando]

    if not operarios_disponibles:
        return asignacion

    # Mapeo estratégico guiado por tus referencias para priorizar la distribución óptima
    mapeo_referencia = {
        "917": "Operario 4",
        "924": "Operario 6"
    }

    # Asignación forzada basada en hitos dedicados al 100% (917 y 924)
    for m in list(maquinas_por_assignar):
        if m in mapeo_referencia:
            op_destino = mapeo_referencia[m]
            if op_destino in operarios_disponibles:
                asignacion[op_destino].append(m)
                maquinas_por_assignar.remove(m)

    # Ordenar el resto de máquinas de mayor a menor carga
    maquinas_por_assignar.sort(key=lambda x: -WORKLOAD_MAESTRO.get(x, 0))

    # Lista de operarios disponibles que restan para asignación general
    ops_pool = [o for o in operarios_disponibles if o not in ["Operario 4", "Operario 6"]]
    if not ops_pool:
        ops_pool = [o for o in operarios_disponibles]

    # Distribución equilibrada considerando proximidad física y el nuevo techo del 110%
    for m in list(maquinas_por_assignar):
        mejor_op = None
        menor_distancia_op = float('inf')
        
        for op in ops_pool:
            maqs_del_op = asignacion[op]
            carga_actual = sum([WORKLOAD_MAESTRO[x] for x in maqs_del_op])
            
            todas_en_pasillo = all(x in HEURISTICA_PASILLO for x in maqs_del_op + [m])
            # Excepción por pasillo ampliada proporcionalmente si corresponde
            tope_limite = 1.35 if todas_en_pasillo else MAX_SATURACION_ESTANDAR
            
            if carga_actual + WORKLOAD_MAESTRO[m] <= tope_limite:
                if maqs_del_op:
                    dist_eval = np.mean([MATRIZ_DISTANCIAS[m].get(ya, 50.0) for ya in maqs_del_op])
                else:
                    dist_eval = 0.0
                
                if maqs_del_op and any(MATRIZ_DISTANCIAS[m].get(ya, 0) > DISTANCIA_CRITICA_MAX for ya in maqs_del_op):
                    continue

                if dist_eval < menor_distancia_op:
                    menor_distancia_op = dist_eval
                    mejor_op = op
                    
        if mejor_op:
            asignacion[mejor_op].append(m)
            maquinas_por_assignar.remove(m)

    # Desborde controlado final para asegurar que ninguna celda activa quede huérfana
    for m in list(maquinas_por_assignar):
        mejor_op_desborde = None
        menor_distancia_desborde = float('inf')
        
        for op in operarios_disponibles:
            maqs_del_op = asignacion[op]
            if maqs_del_op:
                dist_eval = np.mean([MATRIZ_DISTANCIAS[m].get(ya, 50.0) for ya in maqs_del_op])
            else:
                dist_eval = 0.0
                
            if dist_eval < menor_distancia_desborde:
                menor_distancia_desborde = dist_eval
                mejor_op_desborde = op
                
        if mejor_op_desborde:
            asignacion[mejor_op_desborde].append(m)
            maquinas_por_assignar.remove(m)

    return asignacion

# -------------------------------------------------------------------------
# 3. CONTROL DE SESIÓN Y BLINDAJE ANTI-CONGELAMIENTO
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador de Turnos medmix")

if "estados_maquinas" not in st.session_state or not isinstance(st.session_state.estados_maquinas, dict):
    st.session_state.estados_maquinas = {m: "Trabajando" for m in WORKLOAD_MAESTRO.keys()}
    # Ajuste inicial basado en tus capturas activas (Celdas por defecto paradas)
    for desactiva in ["904", "916", "925", "926"]:
        st.session_state.estados_maquinas[desactiva] = "Día Libre"

if "estados_operarios" not in st.session_state or not isinstance(st.session_state.estados_operarios, dict):
    st.session_state.estados_operarios = {op: "Disponible" for op in LISTA_7_OPERARIOS}

if "prioridades_estrellas" not in st.session_state:
    st.session_state.prioridades_estrellas = {m: "⭐⭐ Media" for m in WORKLOAD_MAESTRO.keys()}

maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]

if "propuesta_actual" not in st.session_state or not isinstance(st.session_state.propuesta_actual, dict):
    st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos)

# --- PANEL DE CONTROL LATERAL ---
with st.sidebar:
    st.image("https://www.medmix.mixpac.com/images/medmix_Logo_Pos_RGB.svg", width=180)
    
    st.markdown("### 👤 Disponibilidad de Personal")
    for op in LISTA_7_OPERARIOS:
        estado_previo = st.session_state.estados_operarios.get(op, "Disponible")
        sel_op = st.selectbox(f"{op}:", options=["Disponible", "Día Libre / Ausente"], index=0 if estado_previo == "Disponible" else 1, key=f"sel_status_{op}")
        if sel_op != estado_previo:
            st.session_state.estados_operarios[op] = sel_op
            maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
            ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]
            st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos)
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Estado de Celdas (Activas)")
    for m in sorted(WORKLOAD_MAESTRO.keys()):
        estado_actual = st.session_state.estados_maquinas.get(m, "Trabajando")
        st.markdown(f"**Celda {m}** — Carga: {WORKLOAD_MAESTRO[m]*100:.1f}%")
        
        c_tr, c_dl = st.columns(2)
        with c_tr:
            if st.button("🟢 Activa", key=f"btn_tr_{m}", use_container_width=True, disabled=(estado_actual == "Trabajando")):
                st.session_state.estados_maquinas[m] = "Trabajando"
                maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
                ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]
                st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos)
                st.rerun()
        with c_dl:
            if st.button("🔴 Parada", key=f"btn_dl_{m}", use_container_width=True, disabled=(estado_actual == "Día Libre")):
                st.session_state.estados_maquinas[m] = "Día Libre"
                for op in LISTA_7_OPERARIOS:
                    if m in st.session_state.propuesta_actual[op]:
                        st.session_state.propuesta_actual[op].remove(m)
                maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
                ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]
                st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos)
                st.rerun()

# -------------------------------------------------------------------------
# 4. TABLERO CENTRAL DE KPIs
# -------------------------------------------------------------------------
st.title("🏭 Planificador y Balanceo Dinámico de Celdas")

num_maquinas_trabajando = len(maquinas_activas)
num_operarios_disponibles = len(ops_activos)

cargas_reales_operarios = []
for op in ops_activos:
    maqs_del_op = st.session_state.propuesta_actual.get(op, [])
    cargas_reales_operarios.append(sum([WORKLOAD_MAESTRO.get(x, 0) for x in maqs_del_op]))
saturacion_media_turno = (np.mean(cargas_reales_operarios) * 100) if cargas_reales_operarios else 0.0

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric(label="📊 Nº de Máquinas Trabajando", value=f"{num_maquinas_trabajando} Celdas")
with kpi2:
    st.metric(label="👤 Nº de Operarios Activos", value=f"{num_operarios_disponibles} de 7")
with kpi3:
    st.metric(label="⚡ Saturación Media del Turno", value=f"{saturacion_media_turno:.1f}%")

todas_las_maquinas_en_uso = []
for m_list in st.session_state.propuesta_actual.values():
    todas_las_maquinas_en_uso.extend(m_list)
maquinas_faltantes = set(maquinas_activas) - set(todas_las_maquinas_en_uso)

if maquinas_faltantes:
    st.error(f"⚠️ **ATENCIÓN:** Hay celdas trabajando sin operario asignado: {', '.join(sorted(maquinas_faltantes))}")
else:
    st.success("✅ Cobertura Correcta: El 100% de las celdas activas tienen un operario a cargo.")

st.markdown("---")

# -------------------------------------------------------------------------
# 5. MATRIZ DE FICHAS CON FILTRO DE EXCLUSIÓN CRUZADA (BLOQUEO MANUAL)
# -------------------------------------------------------------------------
st.subheader("🚀 Asignación del Turno (Modificación Manual Protegida)")
cols_res = st.columns(4)

for idx, operario in enumerate(LISTA_7_OPERARIOS):
    esta_disponible = st.session_state.estados_operarios.get(operario, "Disponible") == "Disponible"
    maquinas_del_operario = st.session_state.propuesta_actual.get(operario, [])
    
    with cols_res[idx % 4]:
        if not esta_disponible:
            with st.container(border=True):
                st.markdown(f"<h3 style='color: #888;'>👤 {operario}</h3>", unsafe_allow_html=True)
                st.caption("❌ **Día Libre / Ausente**")
        else:
            with st.container(border=True):
                st.markdown(f"### 👤 {operario}")
                
                maquinas_ocupadas_por_otros = []
                for op_ref, maqs_ref in st.session_state.propuesta_actual.items():
                    if op_ref != operario:
                        maquinas_ocupadas_por_otros.extend(maqs_ref)
                
                opciones_libres = sorted(list(set(maquinas_activas) - set(maquinas_ocupadas_por_otros)))
                opciones_visibles = sorted(list(set(opciones_libres) | set(maquinas_del_operario)))

                nuevas_maquinas = st.multiselect("Asignar celdas:", options=opciones_visibles, default=maquinas_del_operario, key=f"ms_tarjeta_{operario}")
                
                if nuevas_maquinas != maquinas_del_operario:
                    st.session_state.propuesta_actual[operario] = nuevas_maquinas
                    st.rerun()
                
                st.markdown("**Detalle de Cargas:**")
                if nuevas_maquinas:
                    for m in nuevas_maquinas:
                        st.write(f"• **Máquina {m}**: {WORKLOAD_MAESTRO.get(m, 0)*100:.1f}% carga")
                else:
                    st.caption("💤 Sin celdas asignadas.")

                carga_real = sum([WORKLOAD_MAESTRO.get(m, 0) for m in nuevas_maquinas])
                sat_p = carga_real * 100
                
                aplica_excepcion_pasillo = len(nuevas_maquinas) > 0 and all(m in HEURISTICA_PASILLO for m in nuevas_maquinas)
                tope_limite = 135.0 if aplica_excepcion_pasillo else MAX_SATURACION_ESTANDAR
                
                # Alertas visuales actualizadas al nuevo techo (110%)
                if sat_p > 110.0:
                    st.error(f"🔴 Sobrecarga Crítica: {sat_p:.1f}% (Máx 110%)")
                elif sat_p > 97.0:
                    st.warning(f"⚠️ Carga Elevada: {sat_p:.1f}%")
                elif sat_p == 0:
                    pass
                else:
                    st.success(f"⚡ Carga Óptima: {sat_p:.1f}%")

                if len(nuevas_maquinas) > 1:
                    distancias_texto = []
                    alerta_distancia = False
                    for i in range(len(nuevas_maquinas)):
                        for j in range(i + 1, len(nuevas_maquinas)):
                            m1, m2 = nuevas_maquinas[i], nuevas_maquinas[j]
                            dist = MATRIZ_DISTANCIAS.get(m1, {}).get(m2, 0)
                            distancias_texto.append(f"{m1} ↔️ {m2}: {dist} metros")
                            if dist > DISTANCIA_CRITICA_MAX:
                                alerta_distancia = True
                                
                    with st.expander("📍 Verificación de Trayectos", expanded=alerta_distancia):
                        for txt in distancias_texto:
                            if any(f" {x} metros" in txt for x in ["21","23","24","25","26","27","28","29","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","59","62","66","70","74","78","80"]):
                                st.write(f"❌ {txt} — **Inviable**")
                            else:
                                st.write(f"✅ {txt}")

                if nuevas_maquinas:
                    st.write("**Criticidad (Hitos):**")
                    for m in nuevas_maquinas:
                        c1, c2 = st.columns([1, 2])
                        with c1: st.caption(f"🤖 **M-{m}**")
                        with c2:
                            prio_estrella = st.selectbox(f"Prio_{operario}_{m}", options=["⭐⭐⭐ Alta", "⭐⭐ Media", "⭐ Baja"], index=["⭐⭐⭐ Alta", "⭐⭐ Media", "⭐ Baja"].index(st.session_state.prioridades_estrellas.get(m, "⭐⭐ Media")), label_visibility="collapsed", key=f"star_sel_{operario}_{m}")
                            st.session_state.prioridades_estrellas[m] = prio_estrella

# -------------------------------------------------------------------------
# 6. RECALCULO DE IA ASOCIADO FIELMENTE A LOS FILTROS ACTIVOS
# -------------------------------------------------------------------------
st.write("---")
if st.button("🔄 Recalcular por Proximidad Física Real (IA)", type="primary", use_container_width=True):
    st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos)
    st.rerun()
