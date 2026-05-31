import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# -------------------------------------------------------------------------
# 1. BASE DE DATOS MAESTRA Y PARÁMETROS ERGONÓMICOS (MEDMIX)
# -------------------------------------------------------------------------
MAX_SATURACION_ESTANDAR = 1.10  
VELOCIDAD_CAMINADO = 1.2  
FRECUENCIA_TRASLADOS_HORA = 4  

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
# 2. FUNCIÓN DE CÁLCULO DE TRASLADO DINÁMICO (Walking Workload)
# -------------------------------------------------------------------------
def calcular_carga_caminado(lista_maquinas):
    if not lista_maquinas or len(lista_maquinas) <= 1:
        return 0.0
    
    distancias = []
    for i in range(len(lista_maquinas)):
        for j in range(i + 1, len(lista_maquinas)):
            m1, m2 = lista_maquinas[i], lista_maquinas[j]
            distancias.append(MATRIZ_DISTANCIAS.get(m1, {}).get(m2, 25.0))
            
    distancia_media = np.mean(distancias)
    segundos_traslado_hora = (distancia_media / VELOCIDAD_CAMINADO) * FRECUENCIA_TRASLADOS_HORA
    porcentaje_jornada_consumido = segundos_traslado_hora / 3600.0
    return porcentaje_jornada_consumido

# -------------------------------------------------------------------------
# 3. MOTOR IA CON INTEGRACIÓN DE RESTRICCIONES GEOGRÁFICAS Y OPERATIVAS
# -------------------------------------------------------------------------
def optimizar_con_operarios_fijos(maquinas_trabajando, operarios_disponibles, cargas_activas, variante_902):
    asignacion = {op: [] for op in LISTA_8_OPERARIOS}
    maquinas_por_asignar = [m for m in maquinas_trabajando]

    if not operarios_disponibles:
        return asignacion

    # Regla Hito 1: Celdas estructurales de carga completa (100%)
    mapeo_estricto = {"917": "Operario 4", "924": "Operario 6", "928": "Operario 7"}
    for m, op in mapeo_estricto.items():
        if m in maquinas_por_asignar and op in operarios_disponibles:
            asignacion[op].append(m)
            maquinas_por_asignar.remove(m)

    # Regla Hito 2: Célula Operario 1 (927 prioritaria + 902 SÓLO si es Cánula Corta)
    if "Operario 1" in operarios_disponibles:
        if "927" in maquinas_por_asignar:
            asignacion["Operario 1"].append("927")
            maquinas_por_asignar.remove("927")
        if variante_902 == "Cánula Corta (37.1%)" and "902" in maquinas_por_asignar:
            asignacion["Operario 1"].append("902")
            maquinas_por_asignar.remove("902")

    # Regla Hito 3: Compactación Anti-Subutilización en Operario 8 (Bloque Fijo 916 + 904)
    if "Operario 8" in operarios_disponibles:
        for m in ["916", "904"]:
            if m in maquinas_por_asignar:
                asignacion["Operario 8"].append(m)
                maquinas_por_asignar.remove(m)

    # Regla Hito 4: Balanceo Especial Cánula Larga -> Desplazar M-902 al Operario 5
    if variante_902 == "Cánula Larga (45.0%)" and "902" in maquinas_por_asignar:
        if "Operario 5" in operarios_disponibles:
            asignacion["Operario 5"].append("902")
            maquinas_por_asignar.remove("902")

    # Regla Hito 5: Bloques de Pasillos Estándar
    if "Operario 2" in operarios_disponibles:
        for m in ["922", "911", "905"]:
            if m in maquinas_por_asignar:
                asignacion["Operario 2"].append(m)
                maquinas_por_asignar.remove(m)

    if "Operario 3" in operarios_disponibles:
        for m in ["906", "907", "903"]:
            if m in maquinas_por_asignar:
                asignacion["Operario 3"].append(m)
                maquinas_por_asignar.remove(m)

    maquinas_por_asignar.sort(key=lambda x: -cargas_activas.get(x, 0))

    # Reparto Dinámico Seguro por Proximidad Física Real
    for m in list(maquinas_por_asignar):
        mejor_op = None
        menor_carga_total = float('inf')
        
        for op in operarios_disponibles:
            if m == "904" and "928" in asignacion[op]: continue
            if m == "928" and "904" in asignacion[op]: continue
            if m == "902" and variante_902 == "Cánula Larga (45.0%)" and op == "Operario 1": continue

            maqs_del_op = asignacion[op] + [m]
            carga_estatica = sum([cargas_activas[x] for x in maqs_del_op])
            carga_dinamica_pasos = calcular_carga_caminado(maqs_del_op)
            carga_total_proyectada = carga_estatica + carga_dinamica_pasos
            
            if carga_total_proyectada <= MAX_SATURACION_ESTANDAR + 0.05:
                if carga_total_proyectada < menor_carga_total:
                    menor_carga_total = carga_total_proyectada
                    mejor_op = op
                    
        if mejor_op:
            asignacion[mejor_op].append(m)
            maquinas_por_asignar.remove(m)

    for m in list(maquinas_por_asignar):
        ops_validos = [o for o in operarios_disponibles if not (m == "904" and "928" in asignacion[o]) and not (m == "928" and "904" in asignacion[o])]
        if not ops_validos: 
            ops_validos = list(operarios_disponibles)
        
        op_menos_cargado = min(ops_validos, key=lambda o: sum([cargas_activas[x] for x in asignacion[o]]) + calcular_carga_caminado(asignacion[o]))
        asignacion[op_menos_cargado].append(m)

    return asignacion

# -------------------------------------------------------------------------
# 4. CONFIGURACIÓN DE LA INTERFAZ DE USUARIO (RESTABLECIDA AL 100%)
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador de Cargas medmix")

if "version_902" not in st.session_state:
    st.session_state.version_902 = "Cánula Corta (37.1%)"

cargas_dinamicas_turno = WORKLOAD_MAESTRO_BASE.copy()
cargas_dinamicas_turno["902"] = 0.4500 if st.session_state.version_902 == "Cánula Larga (45.0%)" else 0.3712

if "estados_maquinas" not in st.session_state:
    st.session_state.estados_maquinas = {m: "Trabajando" for m in cargas_dinamicas_turno.keys()}
    for desactiva in ["904", "916", "925", "926"]:
        st.session_state.estados_maquinas[desactiva] = "Día Libre"

if "estados_operarios" not in st.session_state:
    st.session_state.estados_operarios = {op: "Disponible" for op in LISTA_8_OPERARIOS}

# --- CONTROL LATERAL ORIGINAL (SELECCIÓN DIRECTA DE BOTONES CON LUZ ALTA) ---
with st.sidebar:
    st.image("https://www.medmix.mixpac.com/images/medmix_Logo_Pos_RGB.svg", width=180)
    st.markdown("### 🏃 Ergonomía: **1.2 m/s**")
    
    st.markdown("---")
    st.markdown("### ⚙️ Mezcla de Productos")
    version_sel = st.selectbox("M-902 - Tipo de Cánula:", options=["Cánula Corta (37.1%)", "Cánula Larga (45.0%)"], index=0 if st.session_state.version_902 == "Cánula Corta (37.1%)" else 1)
    if version_sel != st.session_state.version_902:
        st.session_state.version_902 = version_sel
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Estado de Celdas (Activas)")
    maquinas_ordenadas = sorted(list(cargas_dinamicas_turno.keys()))
    
    for m in maquinas_ordenadas:
        est_actual = st.session_state.estados_maquinas.get(m, "Trabajando")
        st.markdown(f"**Celda {m}** — Carga: {cargas_dinamicas_turno[m]*100:.1f}%")
        col_act, col_par = st.columns(2)
        
        # Lógica de resaltado Poka-Yoke corregida sin causar recargas infinitas
        with col_act:
            if st.button(f"🟢 Activa", key=f"btn_act_{m}", use_container_width=True, type="primary" if est_actual == "Trabajando" else "secondary"):
                st.session_state.estados_maquinas[m] = "Trabajando"
                st.rerun()
        with col_par:
            if st.button(f"🔴 Parada", key=f"btn_par_{m}", use_container_width=True, type="primary" if est_actual == "Día Libre" else "secondary"):
                st.session_state.estados_maquinas[m] = "Día Libre"
                st.rerun()

    st.markdown("---")
    st.markdown("### 👤 Control de Asistencia")
    for op in LISTA_8_OPERARIOS:
        estado_previo = st.session_state.estados_operarios.get(op, "Disponible")
        sel_op = st.selectbox(f"{op}:", options=["Disponible", "Día Libre"], index=0 if estado_previo == "Disponible" else 1, key=f"s_{op}")
        if sel_op != estado_previo:
            st.session_state.estados_operarios[op] = sel_op
            st.rerun()

# Lectura directa de estados de controles laterales
maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]

st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos, cargas_dinamicas_turno, st.session_state.version_902)

# -------------------------------------------------------------------------
# 5. RESUMEN SUPERIOR RECUPERADO (KPI INDICADORES)
# -------------------------------------------------------------------------
# Calcular métricas globales reales
total_celdas_num = len(maquinas_activas)
total_ops_num = len(ops_activos)
suma_cargas = sum([cargas_dinamicas_turno[m] for m in maquinas_activas])
carga_media_global = (suma_cargas / total_ops_num * 100) if total_ops_num > 0 else 0.0

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
with col_kpi1:
    st.metric(label="🔢 Celdas en Producción", value=f"{total_celdas_num} / 16")
with col_kpi2:
    st.metric(label="👥 Operarios Activos", value=f"{total_ops_num} de 8")
with col_kpi3:
    st.metric(label="📊 Carga Media de Saturación", value=f"{carga_media_global:.1f}%")

# -------------------------------------------------------------------------
# 6. CUADRÍCULA CENTRAL DE DISTRIBUCIÓN
# -------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Plan de Cargas con Cálculo de Desplazamiento ($1.2 \\text{ m/s}$)")

todas_las_maquinas_en_uso = []
for op_k in ops_activos: 
    todas_las_maquinas_en_uso.extend(st.session_state.propuesta_actual.get(op_k, []))
maquinas_faltantes = set(maquinas_activas) - set(todas_las_maquinas_en_uso)

if maquinas_faltantes:
    st.error(f"⚠️ ATENCIÓN: Hay celdas trabajando sin operario asignado: {', '.join(sorted(maquinas_faltantes))}")
else:
    st.success("✅ Estabilidad de Línea: Todas las celdas cubiertas correctamente.")

cols_res = st.columns(4)
for idx, operario in enumerate(LISTA_8_OPERARIOS):
    esta_disponible = st.session_state.estados_operarios.get(operario, "Disponible") == "Disponible"
    maquinas_del_operario = st.session_state.propuesta_actual.get(operario, [])
    
    with cols_res[idx % 4]:
        with st.container(border=True):
            st.markdown(f"### 👤 {operario}")
            if not esta_disponible:
                st.markdown("<span style='color:grey; font-style:italic;'>❌ Ausente / Licencia</span>", unsafe_allow_html=True)
            else:
                maquinas_ocupadas_por_otros = []
                for op_ref, maqs_ref in st.session_state.propuesta_actual.items():
                    if op_ref != operario and st.session_state.estados_operarios.get(op_ref) == "Disponible":
                        maquinas_ocupadas_por_otros.extend(maqs_ref)
                
                opciones_libres = sorted(list(set(maquinas_activas) - set(maquinas_ocupadas_por_otros)))
                opciones_visibles = sorted(list(set(opciones_libres) | set(maquinas_del_operario)))

                nuevas_maquinas = st.multiselect(f"Asignar celdas:", options=opciones_visibles, default=maquinas_del_operario, key=f"ms_{operario}")
                
                if nuevas_maquinas != maquinas_del_operario:
                    st.session_state.propuesta_actual[operario] = nuevas_maquinas
                    st.rerun()

                carga_estatica = sum([cargas_dinamicas_turno.get(m, 0.0) for m in nuevas_maquinas]) if nuevas_maquinas else 0.0
                carga_desplazamiento = calcular_carga_caminado(nuevas_maquinas) if nuevas_maquinas else 0.0
                
                carga_total_real = (carga_estatica + carga_desplazamiento) * 100.0

                st.markdown(f"**Carga de Máquinas:** {carga_estatica*100:.1f}%")
                st.markdown("**Carga por Traslado ($1.2 \\text{ m/s}$):** " + f"{carga_desplazamiento*100:.1f}%")
                
                if carga_total_real > 110.0:
                    st.error(f"💥 Carga Total Real: {carga_total_real:.1f}%")
                elif carga_total_real > 95.0:
                    st.warning(f"⚠️ Carga Total Real: {carga_total_real:.1f}%")
                elif carga_total_real == 0.0:
                    st.info("Sin carga asignada")
                else:
                    st.success(f"⚡ Carga Total Real: {carga_total_real:.1f}%")

                if "904" in nuevas_maquinas and "928" in nuevas_maquinas:
                    st.error("🚨 CRÍTICO: Combinación 928+904 prohibida por distancia extrema.")
                if operario == "Operario 1" and "902" in nuevas_maquinas and "927" in nuevas_maquinas and st.session_state.version_902 == "Cánula Larga (45.0%)":
                    st.error("🚨 CRÍTICO: Prohibido juntar M-927 + M-902 en Cánula Larga.")

# -------------------------------------------------------------------------
# 7. BOTONERA INFERIOR (CÁLCULO E IMPRESIÓN INSTANTÁNEA)
# -------------------------------------------------------------------------
st.write("---")
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🔄 Recalcular Optimización por Proximidad Real (IA)", type="primary", use_container_width=True):
        st.session_state.propuesta_actual = optimizar_con_operarios_fijos(maquinas_activas, ops_activos, cargas_dinamicas_turno, st.session_state.version_902)
        st.rerun()

with col_btn2:
    components.html("""
        <button style="
            background-color: #24a0ed;
            border: none;
            color: white;
            padding: 11px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 0px;
            cursor: pointer;
            width: 100%;
            border-radius: 4px;
            font-family: sans-serif;
            font-weight: bold;
        " onclick="window.parent.print()">🖨️ Imprimir / Exportar Reporte de Turno</button>
    """, height=50)
