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
# 2. CÁLCULO DE ERGONOMÍA Y TRASLADOS
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
    return segundos_traslado_hora / 3600.0

# -------------------------------------------------------------------------
# 3. MOTOR IA DE BALANCEO GEOGRÁFICO
# -------------------------------------------------------------------------
def optimizar_con_operarios_fijos(maquinas_trabajando, operarios_disponibles, cargas_activas, variante_902):
    asignacion = {op: [] for op in LISTA_8_OPERARIOS}
    maquinas_por_assignar = [m for m in maquinas_trabajando]

    if not operarios_disponibles:
        return asignacion

    # Hito 1: Asignaciones estructurales rígidas
    mapeo_estricto = {"917": "Operario 4", "924": "Operario 6", "928": "Operario 7"}
    for m, op in mapeo_estricto.items():
        if m in maquinas_por_assignar and op in operarios_disponibles:
            asignacion[op].append(m)
            maquinas_por_assignar.remove(m)

    # Hito 2: Asignación Operario 1
    if "Operario 1" in operarios_disponibles:
        if "927" in maquinas_por_assignar:
            asignacion["Operario 1"].append("927")
            maquinas_por_assignar.remove("927")
        if variante_902 == "Cánula Corta (37.1%)" and "902" in maquinas_por_assignar:
            asignacion["Operario 1"].append("902")
            maquinas_por_assignar.remove("902")

    # Hito 3: Bloque Fijo Operario 8 (Asegurando asignación de 916 y 904)
    if "Operario 8" in operarios_disponibles:
        for m in ["916", "904"]:
            if m in maquinas_por_assignar:
                asignacion["Operario 8"].append(m)
                maquinas_por_assignar.remove(m)

    # Hito 4: Ajuste por Cánula Larga M-902
    if variante_902 == "Cánula Larga (45.0%)" and "902" in maquinas_por_assignar:
        if "Operario 5" in operarios_disponibles:
            asignacion["Operario 5"].append("902")
            maquinas_por_assignar.remove("902")

    # Hito 5: Distribución de Pasillos Estándar
    if "Operario 2" in operarios_disponibles:
        for m in ["922", "911", "905"]:
            if m in maquinas_por_assignar:
                asignacion["Operario 2"].append(m)
                maquinas_por_assignar.remove(m)

    if "Operario 3" in operarios_disponibles:
        for m in ["906", "907", "903"]:
            if m in maquinas_por_assignar:
                asignacion["Operario 3"].append(m)
                maquinas_por_assignar.remove(m)

    maquinas_por_assignar.sort(key=lambda x: -cargas_activas.get(x, 0))

    # Distribución inteligente dinámica
    for m in list(maquinas_por_assignar):
        mejor_op = None
        menor_carga_total = float('inf')
        
        for op in operarios_disponibles:
            if m == "904" and "928" in asignacion[op]: continue
            if m == "928" and "904" in asignacion[op]: continue
            if m == "902" and variante_902 == "Cánula Larga (45.0%)" and op == "Operario 1": continue

            maqs_del_op = asignacion[op] + [m]
            carga_total_proyectada = sum([cargas_activas[x] for x in maqs_del_op]) + calcular_carga_caminado(maqs_del_op)
            
            if carga_total_proyectada <= MAX_SATURACION_ESTANDAR + 0.05:
                if carga_total_proyectada < menor_carga_total:
                    menor_carga_total = carga_total_proyectada
                    mejor_op = op
                    
        if mejor_op:
            asignacion[mejor_op].append(m)
            maquinas_por_assignar.remove(m)

    # Forzar asignación limpia para remanentes
    for m in list(maquinas_por_assignar):
        ops_validos = [o for o in operarios_disponibles if not (m == "904" and "928" in asignacion[o]) and not (m == "928" and "904" in asignacion[o])]
        if not ops_validos: ops_validos = list(operarios_disponibles)
        op_menos_cargado = min(ops_validos, key=lambda o: sum([cargas_activas[x] for x in asignacion[o]]) + calcular_carga_caminado(asignacion[o]))
        asignacion[op_menos_cargado].append(m)

    return asignacion

# -------------------------------------------------------------------------
# 4. INICIALIZACIÓN DEL SISTEMA
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador de Cargas medmix")

if "version_902" not in st.session_state:
    st.session_state.version_902 = "Cánula Corta (37.1%)"

cargas_dinamicas_turno = WORKLOAD_MAESTRO_BASE.copy()
cargas_dinamicas_turno["902"] = 0.4500 if st.session_state.version_902 == "Cánula Larga (45.0%)" else 0.3712

# Inicialización corregida: 904 y 916 por defecto están Activas ("Trabajando")
if "estados_maquinas" not in st.session_state:
    st.session_state.estados_maquinas = {m: "Trabajando" for m in cargas_dinamicas_turno.keys()}
    for desactiva in ["925", "926"]:  # Dejamos activas las indicadas
        st.session_state.estados_maquinas[desactiva] = "Día Libre"

if "estados_operarios" not in st.session_state:
    st.session_state.estados_operarios = {op: "Disponible" for op in LISTA_8_OPERARIOS}

maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]

# Inicializar mapa interno limpio para evitar mutaciones prohibidas de Streamlit
if "mapa_asignaciones" not in st.session_state:
    st.session_state.mapa_asignaciones = optimizar_con_operarios_fijos(maquinas_activas, ops_activos, cargas_dinamicas_turno, st.session_state.version_902)

# -------------------------------------------------------------------------
# 5. CONTROLES EN PANEL LATERAL
# -------------------------------------------------------------------------
with st.sidebar:
    st.image("https://www.medmix.mixpac.com/images/medmix_Logo_Pos_RGB.svg", width=180)
    st.markdown("### 🏃 Ergonomía: **1.2 m/s**")
    
    st.markdown("---")
    st.markdown("### ⚙️ Mezcla de Productos")
    version_sel = st.selectbox("M-902 - Tipo de Cánula:", options=["Cánula Corta (37.1%)", "Cánula Larga (45.0%)"], index=0 if st.session_state.version_902 == "Cánula Corta (37.1%)" else 1)
    if version_sel != st.session_state.version_902:
        st.session_state.version_902 = version_sel
        st.session_state.mapa_asignaciones = optimizar_con_operarios_fijos(maquinas_activas, ops_activos, cargas_dinamicas_turno, version_sel)
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Estado de Celdas (Activas)")
    for m in sorted(list(cargas_dinamicas_turno.keys())):
        est_actual = st.session_state.estados_maquinas.get(m, "Trabajando")
        st.markdown(f"**Celda {m}** — Carga: {cargas_dinamicas_turno[m]*100:.1f}%")
        col_act, col_par = st.columns(2)
        
        with col_act:
            if st.button(f"🟢 Activa", key=f"btn_act_{m}", use_container_width=True, type="primary" if est_actual == "Trabajando" else "secondary"):
                st.session_state.estados_maquinas[m] = "Trabajando"
                maquinas_activas_update = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
                st.session_state.mapa_asignaciones = optimizar_con_operarios_fijos(maquinas_activas_update, ops_activos, cargas_dinamicas_turno, st.session_state.version_902)
                st.rerun()
        with col_par:
            if st.button(f"🔴 Parada", key=f"btn_par_{m}", use_container_width=True, type="primary" if est_actual == "Día Libre" else "secondary"):
                st.session_state.estados_maquinas[m] = "Día Libre"
                # Limpiar de las asignaciones actuales
                for op_k in LISTA_8_OPERARIOS:
                    if m in st.session_state.mapa_asignaciones.get(op_k, []):
                        st.session_state.mapa_asignaciones[op_k].remove(m)
                st.rerun()

    st.markdown("---")
    st.markdown("### 👤 Control de Asistencia")
    for op in LISTA_8_OPERARIOS:
        estado_previo = st.session_state.estados_operarios.get(op, "Disponible")
        sel_op = st.selectbox(f"{op}:", options=["Disponible", "Día Libre"], index=0 if estado_previo == "Disponible" else 1, key=f"s_{op}")
        if sel_op != estado_previo:
            st.session_state.estados_operarios[op] = sel_op
            if sel_op == "Día Libre":
                st.session_state.mapa_asignaciones[op] = []
            st.rerun()

# Sincronizar listas actuales de celdas y operarios activos
maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]

# -------------------------------------------------------------------------
# 6. RESUMEN KPI SUPERIOR
# -------------------------------------------------------------------------
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
# 7. CUADRÍCULA CENTRAL DE TARJETAS DE TRABAJO
# -------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Plan de Cargas con Cálculo de Desplazamiento ($1.2 \\text{ m/s}$)")

# Alerta de Pokayoke en tiempo real
maquinas_asignadas_en_pantalla = []
for op in ops_activos:
    maquinas_asignadas_en_pantalla.extend(st.session_state.mapa_asignaciones.get(op, []))

maquinas_faltantes = set(maquinas_activas) - set(maquinas_asignadas_en_pantalla)
if maquinas_faltantes:
    st.error(f"⚠️ ATENCIÓN: Hay celdas trabajando sin operario asignado: {', '.join(sorted(maquinas_faltantes))}")
else:
    st.success("✅ Estabilidad de Línea: Todas las celdas cubiertas correctamente.")

cols_res = st.columns(4)
for idx, operario in enumerate(LISTA_8_OPERARIOS):
    esta_disponible = st.session_state.estados_operarios.get(operario, "Disponible") == "Disponible"
    
    with cols_res[idx % 4]:
        with st.container(border=True):
            st.markdown(f"### 👤 {operario}")
            if not esta_disponible:
                st.markdown("<span style='color:grey; font-style:italic;'>❌ Ausente / Licencia</span>", unsafe_allow_html=True)
            else:
                # Filtrar opciones dinámicas para evitar colisiones manuales entre operarios
                otras_maquinas = []
                for o_ref in ops_activos:
                    if o_ref != operario:
                        otras_maquinas.extend(st.session_state.mapa_asignaciones.get(o_ref, []))
                
                opciones_disponibles = sorted(list(set(maquinas_activas) - set(otras_maquinas)))
                maquinas_actuales_del_op = st.session_state.mapa_asignaciones.get(operario, [])
                opciones_finales = sorted(list(set(opciones_disponibles) | set(maquinas_actuales_del_op)))

                # Cambiamos el enfoque: Usamos el value directo mapeado del diccionario maestro de la sesión
                nuevas_maquinas = st.multiselect(
                    f"Asignar celdas:", 
                    options=opciones_finales, 
                    default=maquinas_actuales_del_op,
                    key=f"ms_render_{operario}"
                )
                
                # Mantener sincronizado el diccionario maestro ante cambios manuales del usuario
                st.session_state.mapa_asignaciones[operario] = nuevas_maquinas

                carga_estatica = sum([cargas_dinamicas_turno.get(m, 0.0) for m in nuevas_maquinas])
                carga_dinamica = calcular_carga_caminado(nuevas_maquinas)
                carga_total_real = (carga_estatica + carga_dinamica) * 100.0

                st.markdown(f"**Carga de Máquinas:** {carga_estatica*100:.1f}%")
                st.markdown("**Carga por Traslado ($1.2 \\text{ m/s}$):** " + f"{carga_dinamica*100:.1f}%")
                
                if carga_total_real > 110.0:
                    st.error(f"💥 Carga Total Real: {carga_total_real:.1f}%")
                elif carga_total_real > 95.0:
                    st.warning(f"⚠️ Carga Total Real: {carga_total_real:.1f}%")
                elif carga_total_real == 0.0:
                    st.info("Sin carga asignada")
                else:
                    st.success(f"⚡ Carga Total Real: {carga_total_real:.1f}%")

                # Alertas operativas críticas de planta
                if "904" in nuevas_maquinas and "928" in nuevas_maquinas:
                    st.error("🚨 CRÍTICO: Combinación 928+904 prohibida por distancia extrema.")
                if operario == "Operario 1" and "902" in nuevas_maquinas and "927" in nuevas_maquinas and st.session_state.version_902 == "Cánula Larga (45.0%)":
                    st.error("🚨 CRÍTICO: Prohibido juntar M-927 + M-902 en Cánula Larga.")

# -------------------------------------------------------------------------
# 8. PANEL INFERIOR DE ACCIONES DIRECTAS
# -------------------------------------------------------------------------
st.write("---")
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🔄 Recalcular Optimización por Proximidad Real (IA)", type="primary", use_container_width=True):
        # Ejecutar el algoritmo e inyectarlo en el diccionario maestro seguro, evitando errores de clave
        st.session_state.mapa_asignaciones = optimizar_con_operarios_fijos(maquinas_activas, ops_activos, cargas_dinamicas_turno, st.session_state.version_902)
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
