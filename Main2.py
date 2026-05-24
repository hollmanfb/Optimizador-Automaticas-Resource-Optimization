import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------------------------------------------------------
# 1. BASE DE DATOS MAESTRA (TIEMPOS, CARGAS Y DISTANCIAS REALES DE PLANTA)
# -------------------------------------------------------------------------
MAX_SATURACION_ESTANDAR = 0.97  
DISTANCIA_CRITICA_MAX = 20.0  # Límite estricto en metros para evitar fatiga y pérdidas por traslado

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
# 2. MOTOR IA DE ASIGNACIÓN CON DOBLE ETAPA (CONFORT + DESBORDE)
# -------------------------------------------------------------------------
def optimizar_con_operarios_fijos(maquinas_trabajando, operarios_disponibles):
    # Inicialización del mapa completo de la planta (los 7 puestos fijos)
    asignacion = {op: [] for op in LISTA_7_OPERARIOS}
    maquinas_por_asignar = [m for m in maquinas_trabajando]

    if not operarios_disponibles:
        return asignacion

    # --- PASO 1: ASIGNACIÓN DE CELDAS CRÍTICAS DEDICADAS (CARGA >= 100%) ---
    ops_pool = [o for o in operarios_disponibles]
    for m in list(maquinas_por_asignar):
        if WORKLOAD_MAESTRO.get(m, 0) >= 1.00:
            if ops_pool:
                op_elegido = ops_pool.pop(0)
                asignacion[op_elegido].append(m)
                maquinas_por_asignar.remove(m)

    # Ordenar el remanente de mayor a menor carga de trabajo para optimizar empaquetado
    maquinas_por_asignar.sort(key=lambda x: -WORKLOAD_MAESTRO.get(x, 0))

    # --- PASO 2: ASIGNACIÓN EFICIENTE (ZONA DE CONFORT < 97%) ---
    for m in list(maquinas_por_asignar):
        mejor_op = None
        menor_distancia_op = float('inf')
        
        for op in operarios_disponibles:
            maqs_del_op = asignacion[op]
            carga_actual = sum([WORKLOAD_MAESTRO[x] for x in maqs_del_op])
            
            # Ajuste de tolerancia por proximidad en pasillo continuo
            todas_en_pasillo = all(x in HEURISTICA_PASILLO for x in maqs_del_op + [m])
            tope_limite = 1.30 if todas_en_pasillo else MAX_SATURACION_ESTANDAR
            
            if carga_actual + WORKLOAD_MAESTRO[m] <= tope_limite:
                # Control estricto de distancias: No asignar si supera los metros permitidos en planta
                if any(MATRIZ_DISTANCIAS.get(m, {}).get(ya, 0) > DISTANCIA_CRITICA_MAX for ya in maqs_del_op):
                    continue
                
                dist_eval = 0.0 if not maqs_del_op else np.mean([MATRIZ_DISTANCIAS[m].get(ya, 50.0) for ya in maqs_del_op])
                if dist_eval < menor_distancia_op:
                    menor_distancia_op = dist_eval
                    mejor_op = op
                    
        if mejor_op:
            asignacion[mejor_op].append(m)
            maquinas_por_asignar.remove(m)

    # --- PASO 3: FASE DE DESBORDE CRÍTICO (SE SUPERA EL 100% PERO SE RESPETAN METROS) ---
    # Si quedan máquinas activas y ya no hay espacio "limpio", las forzamos por cercanía física
    for m in list(maquinas_por_asignar):
        mejor_op_desborde = None
        menor_distancia_desborde = float('inf')
        
        for op in operarios_disponibles:
            maqs_del_op = asignacion[op]
            
            # Se mantiene el candado de distancia absoluta para evitar trayectos imposibles
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
# 3. INTERFAZ GRÁFICA DE USUARIO (STREAMLIT)
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador Semanal medmix")

# Inicialización persistente del estado de la planta
if "estados_maquinas" not in st.session_state:
    st.session_state.estados_maquinas = {m: "Trabajando" for m in WORKLOAD_MAESTRO.keys()}
    # Ajuste inicial estándar (algunas celdas en reposo)
    for desactiva in ["904", "906", "916", "917", "925", "926", "928"]:
        st.session_state.estados_maquinas[desactiva] = "Día Libre"

if "estados_operarios" not in st.session_state:
    # Empezamos el turno con una configuración de 4 operarios disponibles para pruebas de carga
    st.session_state.estados_operarios = {op: "Disponible" if idx < 4 else "Día Libre / Ausente" for idx, op in enumerate(LISTA_7_OPERARIOS)}

if "prioridades_estrellas" not in st.session_state:
    st.session_state.prioridades_estrellas = {m: "⭐⭐ Media" for m in WORKLOAD_MAESTRO.keys()}

# --- CONTROL DE MANDOS IZQUIERDO (SIDEBAR) ---
with st.sidebar:
    st.image("https://www.medmix.mixpac.com/images/medmix_Logo_Pos_RGB.svg", width=180)
    st.subheader("👤 Gestión de Personal (Estructura Fija)")
    st.write("Cambia el estado para recalcular la carga en tiempo real:")
    
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
    st.subheader("⚙️ Panel Operativo de Celdas")
    
    cambio_maquinas = False
    for m in sorted(WORKLOAD_MAESTRO.keys()):
        st.markdown(f"**Máquina {m}** *(Carga: {WORKLOAD_MAESTRO[m]*100:.1f}%)*")
        estado_actual = st.session_state.estados_maquinas.get(m, "Trabajando")
        
        c_tr, c_dl, c_av = st.columns(3)
        with c_tr:
            if st.button("🟢 Trab" if estado_actual == "Trabajando" else "Trab", key=f"btn_tr_{m}", use_container_width=True):
                st.session_state.estados_maquinas[m] = "Trabajando"; cambio_maquinas = True
        with c_dl:
            if st.button("🟡 Libre" if estado_actual == "Día Libre" else "Libre", key=f"btn_dl_{m}", use_container_width=True):
                st.session_state.estados_maquinas[m] = "Día Libre"; cambio_maquinas = True
        with c_av:
            if st.button("🔴 Avería" if estado_actual == "Avería" else "Avería", key=f"btn_av_{m}", use_container_width=True):
                st.session_state.estados_maquinas[m] = "Avería"; cambio_maquinas = True

    if cambio_maquinas or cambio_operarios:
        maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
        ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]
        st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos)
        st.rerun()

maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]

if "propuesta_actual" not in st.session_state:
    st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos)

# -------------------------------------------------------------------------
# 4. CUADRO DE MANDOS CENTRAL Y KPI'S DE CAPACIDAD
# -------------------------------------------------------------------------
st.title("🏭 Planificación y Balanceo de Celdas — Área de Montaje")

num_maquinas_trabajando = len(maquinas_activas)
num_operarios_disponibles = len(ops_activos)

cargas_ocupadas = [sum([WORKLOAD_MAESTRO.get(m, 0) for m in st.session_state.propuesta_actual.get(op, [])]) for op in ops_activos]
saturacion_media_turno = (np.mean(cargas_ocupadas) * 100) if cargas_ocupadas else 0.0

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric(label="📊 Celdas Activas en Planta", value=f"{num_maquinas_trabajando} Máquinas")
with kpi2:
    st.metric(label="👤 Personal Disponible", value=f"{num_operarios_disponibles} de 7")
with kpi3:
    st.metric(label="⚡ Saturación Media del Personal", value=f"{saturacion_media_turno:.1f}%")

# --- CONCILIACIÓN DE PLANTA (VERIFICACIÓN DE HUÉRFANOS) ---
todas_las_maquinas_en_uso = []
for m_list in st.session_state.propuesta_actual.values():
    todas_las_maquinas_en_uso.extend(m_list)
maquinas_faltantes = set(maquinas_activas) - set(todas_las_maquinas_en_uso)

if maquinas_faltantes:
    st.error(f"🚨 **ALERTA CRÍTICA DE CAPACIDAD:** Faltan operarios en el turno. Las celdas **{', '.join(sorted(maquinas_faltantes))}** están operando y no se han podido asignar por restricciones de distancia. ¡Por favor, habilita un nuevo operario en el panel de la izquierda!")
else:
    st.success("✅ Cobertura Completa: Todas las celdas activas tienen un recurso asignado.")

st.markdown("---")

# -------------------------------------------------------------------------
# 5. MATRIZ DE PLANIFICACIÓN (TARJETAS VISUALES)
# -------------------------------------------------------------------------
st.subheader("🚀 Plan del Turno Activo (Modificación Manual Permitida)")

resultado_final_impresion = {}
cols_res = st.columns(4)

for idx, operario in enumerate(LISTA_7_OPERARIOS):
    esta_disponible = st.session_state.estados_operarios.get(operario, "Disponible") == "Disponible"
    maquinas_del_operario = st.session_state.propuesta_actual.get(operario, [])
    
    with cols_res[idx % 4]:
        if not esta_disponible:
            with st.container(border=True):
                st.markdown(f"<h3 style='color: #a0a0a0;'>👤 {operario}</h3>", unsafe_allow_html=True)
                st.caption("❌ **Día Libre / Ausente**")
        else:
            with st.container(border=True):
                st.markdown(f"### 👤 {operario}")
                
                # Exclusión mutua para evitar duplicar celdas en diferentes personas
                otras_asignadas = []
                for op_ref, maqs_ref in st.session_state.propuesta_actual.items():
                    if op_ref != operario:
                        otras_asignadas.extend(maqs_ref)
                        
                opciones_libres = sorted(list(set(maquinas_activas) - set(otras_asignadas)))
                opciones_visibles = sorted(list(set(opciones_libres) | set(maquinas_del_operario)))

                nuevas_maquinas = st.multiselect("Celdas asignadas:", options=opciones_visibles, default=maquinas_del_operario, key=f"ms_tarjeta_{operario}")
                st.session_state.propuesta_actual[operario] = nuevas_maquinas
                
                # Análisis de Tiempos y Saturación
                carga_real = sum([WORKLOAD_MAESTRO.get(m, 0) for m in nuevas_maquinas])
                sat_p = carga_real * 100
                
                aplica_excepcion_pasillo = len(nuevas_maquinas) > 0 and all(m in HEURISTICA_PASILLO for m in nuevas_maquinas)
                tope_limite = 130.0 if aplica_excepcion_pasillo else 97.0
                
                if sat_p > 100.0:
                    st.error(f"🔴 Sobrecarga Crítica: {sat_p:.1f}%")
                elif sat_p > tope_limite:
                    st.warning(f"⚠️ Carga Elevada: {sat_p:.1f}%")
                elif sat_p == 0:
                    st.caption("💤 Puesto Vacante (Sin Carga)")
                else:
                    st.info(f"⚡ Carga Óptima: {sat_p:.1f}%")

                # Auditoría de Distancias de la Tarjeta en Tiempo Real
                if len(nuevas_maquinas) > 1:
                    distancias_texto = []
                    alerta_distancia = False
                    for i in range(len(nuevas_maquinas)):
                        for j in range(i + 1, len(nuevas_maquinas)):
                            m1, m2 = nuevas_maquinas[i], nuevas_maquinas[j]
                            dist = MATRIZ_DISTANCIAS.get(m1, {}).get(m2, 0)
                            distancias_texto.append(f"{m1} ↔️ {m2}: {dist}m")
                            if dist > DISTANCIA_CRITICA_MAX:
                                alerta_distancia = True
                                
                    with st.expander("📍 Verificación de Trayectos", expanded=alerta_distancia):
                        for txt in distancias_texto:
                            if any(f"{x}m" in txt for x in ["21","23","24","25","26","27","28","29","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","59","62","66","70","74","78","80"]):
                                st.write(f"❌ {txt} — **¡Inviable!**")
                            else:
                                st.write(f"✅ {txt}")

                if nuevas_maquinas:
                    st.write("**Criticidad de Atención (Hitos):**")
                    for m in nuevas_maquinas:
                        c1, c2 = st.columns([1, 2])
                        with c1: st.caption(f"🤖 **M-{m}**")
                        with c2:
                            prio_estrella = st.selectbox(f"Prio_{operario}_{m}", options=["⭐⭐⭐ Alta", "⭐⭐ Media", "⭐ Baja"], index=["⭐⭐⭐ Alta", "⭐⭐ Media", "⭐ Baja"].index(st.session_state.prioridades_estrellas.get(m, "⭐⭐ Media")), label_visibility="collapsed", key=f"star_sel_{operario}_{m}")
                            st.session_state.prioridades_estrellas[m] = prio_estrella
                            
                resultado_final_impresion[operario] = {"maquinas": nuevas_maquinas, "carga": sat_p}

# -------------------------------------------------------------------------
# 6. PANEL DE ACCIÓN GENERAL DE PLANTA
# -------------------------------------------------------------------------
st.write("---")
col_acc1, col_acc2 = st.columns([1, 3])
with col_acc1:
    if st.button("🔄 Recalcular por Proximidad (IA)"):
        st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos)
        st.rerun()

# Documentación técnica de salida para el piso de planta
html_print = "<html><body><h2>medmix - Planificación del Turno Oficial</h2></body></html>"
with col_acc2:
    st.download_button(label="🖨️ Descargar Hoja de Ruta para Planta", data=html_print, file_name=f"Plan_Montaje_Medmix.html", mime="text/html")
