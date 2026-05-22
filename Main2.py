# Workload maestro de las máquinas
const WORKLOAD_MAESTRO = {
  "902": 0.3712, "903": 0.3437, "904": 0.3016, "905": 0.3218,
  "906": 0.3289, "907": 0.3217, "911": 0.1821, "916": 0.3868,
  "917": 1.0000, "922": 0.5321, "923": 0.6995, "924": 1.0000,
  "925": 0.3356, "926": 0.3361, "927": 0.5300, "928": 0.6735
};

// Coordenadas aproximadas del layout (X, Y) para calcular cercanía
const LAYOUT = {
  "922": [1, 4], "907": [3, 4], "902": [4.5, 4], "927": [6.5, 4], "904": [9, 4], "916": [9, 3.5],
  "911": [1, 2], "926": [1, 1], "925": [2, 1], "905": [3, 1],
  "903": [4, 1], "924": [6, 1.5], "923": [9, 1]
};

function calcularDistancia(m1, m2) {
  if (LAYOUT[m1] && LAYOUT[m2]) {
    return Math.sqrt(Math.pow(LAYOUT[m1][0] - LAYOUT[m2][0], 2) + Math.pow(LAYOUT[m1][1] - LAYOUT[m2][1], 2));
  }
  return 5.0;
}

function optimizarTurno() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  
  // Supongamos que en las columnas A (Máquina), B (Activa: SI/NO), C (Asignación Manual) están tus inputs
  // Leemos los datos de la hoja (Filas 2 a 15 por ejemplo)
  const rangoDatos = sheet.getRange("A2:C15").getValues();
  
  let maquinasActivas = [];
  let asignacionesManuales = {};
  
  rangoDatos.forEach(fila => {
    let maq = fila[0].toString().trim();
    let activa = fila[1].toString().toUpperCase().trim();
    let manual = fila[2].toString().trim();
    
    if (maq && activa === "SI") {
      maquinasActivas.push(maq);
      if (manual !== "") {
        asignacionesManuales[maq] = manual;
      }
    }
  });
  
  // Algoritmo de asignación efiente
  let maquinasPorAsignar = maquinasActivas.filter(m => !asignacionesManuales[m]);
  // Ordenar por carga descendente para asegurar eficiencia en el empaquetado
  maquinasPorAsignar.sort((a, b) => (WORKLOAD_MAESTRO[b] || 0) - (WORKLOAD_MAESTRO[a] || 0));
  
  let operarios = {};
  
  // Forzar asignaciones manuales primero
  for (let m in asignacionesManuales) {
    let op = asignacionesManuales[m];
    if (!operarios[op]) operarios[op] = { maquinas: [], carga: 0 };
    operarios[op].maquinas.push(m);
    operarios[op].carga += (WORKLOAD_MAESTRO[m] || 0);
  }
  
  let opIdx = 1;
  while (maquinasPorAsignar.length > 0) {
    let opActual = "Operario " + opIdx;
    while (operarios[opActual]) { 
      opIdx++; 
      opActual = "Operario " + opIdx; 
    }
    
    operarios[opActual] = { maquinas: [], carga: 0 };
    let pivote = maquinasPorAsignar.shift();
    operarios[opActual].maquinas.push(pivote);
    operarios[opActual].carga += WORKLOAD_MAESTRO[pivote];
    
    // Buscar la más cercana que quepa por debajo del 97%
    let continuar = true;
    while (continuar) {
      let mejorIdx = -1;
      let minDist = 999;
      
      for (let i = 0; i < maquinasPorAsignar.length; i++) {
        let m = maquinasPorAsignar[i];
        let cargaM = WORKLOAD_MAESTRO[m] || 0;
        
        if (operarios[opActual].carga + cargaM <= 0.97) {
          // Distancia media a las que ya tiene el operario
          let distAcum = 0;
          operarios[opActual].maquinas.forEach(yaAsig => {
            distAcum += calcularDistancia(m, yaAsig);
          });
          let distProm = distAcum / operarios[opActual].maquinas.length;
          
          if (distProm < minDist) {
            minDist = distProm;
            mejorIdx = i;
          }
        }
      }
      
      if (mejorIdx !== -1) {
        let mElegida = maquinasPorAsignar.splice(mejorIdx, 1)[0];
        operarios[opActual].maquinas.push(mElegida);
        operarios[opActual].carga += WORKLOAD_MAESTRO[mElegida];
      } else {
        continuar = false;
      }
    }
    opIdx++;
  }
  
  // Imprimir los resultados de vuelta en la hoja (por ejemplo, en la columna E y F)
  sheet.getRange("E2:F20").clearContent(); // Limpiar salidas anteriores
  let filaSalida = 2;
  sheet.getRange(1, 5).setValue("Operario");
  sheet.getRange(1, 6).setValue("Asignación Máquinas (% Carga)");
  
  for (let op in operarios) {
    if(operarios[op].maquinas.length > 0) {
      let descripcion = operarios[op].maquinas.map(m => `Máq ${m} (${(WORKLOAD_MAESTRO[m]*100).toFixed(1)}%)`).join(" + ");
      sheet.getRange(filaSalida, 5).setValue(op + ` [Tot: ${(operarios[op].carga * 100).toFixed(1)}%]`);
      sheet.getRange(filaSalida, 6).setValue(descripcion);
      filaSalida++;
    }
  }
}
