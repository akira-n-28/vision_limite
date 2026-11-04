/* Logica della revisione offline del campione-2 dei sintetici v4: stato, primo e secondo passaggio, coppie,
   decontaminazione, esportazione e importazione. Nessun accesso al DOM né alla rete: la usano index.html (app.js) e
   verifica.html (test.js). */
(function (root) {
  "use strict";

  var SCHEMA_STATO = "synth-v4-revisione-avanzamento/1";
  var NON = "non_compilato";

  var VALORI = {
    leggibilita: [
      [NON, "— non ancora compilato —"],
      ["si", "sì: l'informazione è visibile, leggibile e univoca"],
      ["no", "no: manca, è coperta, tagliata, illeggibile o ambigua"],
      ["incerto", "incerto: ho guardato ma non so decidere"],
      ["non_giudicabile", "non giudicabile: il caso non permette un giudizio"]
    ],
    giudizio: [
      [NON, "— non ancora compilato —"],
      ["si", "sì"],
      ["no", "no"],
      ["incerto", "incerto: ho guardato ma non so decidere"],
      ["non_giudicabile", "non giudicabile: il caso non permette un giudizio"],
      ["non_verificabile", "non verificabile con il materiale del pacchetto"],
      ["non_applicabile", "non applicabile a questo caso"]
    ],
    gravita: [
      [NON, "— non ancora compilato —"],
      ["nessun_difetto", "nessun difetto"],
      ["minore", "minore: difetto che non cambia risposta né lettura"],
      ["grave", "grave: errore"],
      ["incerto", "incerto"],
      ["non_applicabile", "non applicabile"]
    ],
    esame: [
      ["non_esaminata", "— non ancora esaminata —"],
      ["stesso_contenuto", "stesso contenuto (stessa immagine o stessa scena)"],
      ["contenuto_diverso", "contenuto diverso"],
      ["incerto", "incerto"],
      ["non_giudicabile", "non giudicabile (immagine assente o illeggibile)"]
    ],
    esito_decontaminazione: [
      [NON, "— non ancora compilato —"],
      ["si", "sì: stesso contenuto di un elemento del pannello"],
      ["no", "no: nessuna corrispondenza con il pannello"],
      ["incerto", "incerto"],
      ["non_giudicabile", "non giudicabile"]
    ]
  };

  var AIUTI = {
    risposta: "La tua risposta alla domanda, scritta prima di vedere la soluzione. Puoi usare simboli e più righe.",
    non_riesco: "Spunta se dall'immagine e dal prompt non riesci a rispondere: vale come risposta.",
    leggibilita: "Giudizio iniziale: l'informazione che serve è visibile, leggibile alla risoluzione data e univoca?",
    note_prima: "Facoltative: che cosa ti ha fatto esitare, dove guardare.",
    corretta: "Il target (risposta attesa) è la risposta vera per ciò che si vede e per la scena descritta dalla specifica?",
    coerente_con_annotazioni: "Il PNG mostra ciò che dicono specifica e layout (valori, etichette, posizioni)?",
    rispondibile_da_immagine: "Si risponde con la sola immagine e il prompt, senza informazioni esterne?",
    gravita: "Grave = errore (target sbagliato, non rispondibile, nessuna o più opzioni vere, contratto della coppia " +
             "violato). Minore = difetto che non cambia la risposta né la lettura.",
    tipo_difetto: "Obbligatorio con gravità minore o grave: scegli un tipo o scrivine uno.",
    note_dopo: "Note del giudizio finale; più righe ammesse.",
    risposta_dopo_soluzione: "Facoltativa. La tua risposta dopo aver visto la soluzione: non è indipendente e resta " +
                             "separata da quella iniziale.",
    contratto_coppia: "La variante b cambia la risposta e solo il dettaglio dichiarato rispetto alla a (stessa " +
                      "domanda e stesse opzioni)?",
    esame: "Confronta questa immagine del pannello con il record sintetico.",
    esito_decontaminazione: "Sì solo indicando la corrispondenza decisiva; no solo dopo aver esaminato tutte le " +
                            "corrispondenze come contenuto diverso e con il materiale completo.",
    decisiva: "La corrispondenza che mostra lo stesso contenuto."
  };

  var TIPI_DIFETTO = [
    "target_errato", "domanda_non_rispondibile", "elemento_nascosto_tagliato_o_coperto", "illeggibile_alla_risoluzione",
    "ambiguita", "nessuna_opzione_vera", "piu_opzioni_vere", "immagine_diversa_dalla_specifica",
    "coppia_risposta_invariata", "coppia_piu_dettagli_cambiati", "testi_attaccati_ma_leggibili",
    "etichetta_lontana_ma_univoca", "forma_della_domanda", "altro"
  ];

  var CAMPI_DOPO = ["corretta", "coerente_con_annotazioni", "rispondibile_da_immagine", "gravita"];
  var TSV_ESITI = ["unita", "variante", "record", "risposta_revisore", "corretta", "coerente_con_annotazioni",
                   "rispondibile_da_immagine", "spiegazione", "contratto_coppia", "gravita", "tipo_difetto", "tempo_s",
                   "note"];
  var TSV_DECONTAMINAZIONE = ["unita", "record", "stesso_contenuto", "corrispondenze_guardate",
                              "corrispondenza_decisiva", "note"];

  function ammessi(nome) {
    return VALORI[nome].map(function (x) { return x[0]; });
  }

  function ora() {
    return new Date().toISOString();
  }

  function haValore(lista, valore) {
    return lista.indexOf(valore) >= 0;
  }

  /* ------------------------------------------------------------ indice del pacchetto */

  function indice(pkg) {
    var idx = {unita: {}, decontaminazione: {}, sezioni: {campione: pkg.campione, mirati: pkg.mirati,
                                                          decontaminazione: pkg.decontaminazione}};
    pkg.campione.concat(pkg.mirati).forEach(function (u) { idx.unita[u.id] = u; });
    pkg.decontaminazione.forEach(function (d) {
      var ids = {};
      d.elenco.forEach(function (x) { ids[x.id] = x; });
      idx.decontaminazione[d.id] = {d: d, corrispondenze: ids};
    });
    return idx;
  }

  function nuovoStato(pkg) {
    var t = ora();
    return {schema: SCHEMA_STATO, pacchetto: {id: pkg.id, versione: pkg.versione, impronta: pkg.impronta},
            revisore: "", creato_il: t, aggiornato_il: t, unita: {}, decontaminazione: {}, posizione: null};
  }

  function evento(voce, azione, dettaglio) {
    var e = {quando: ora(), azione: azione};
    if (dettaglio !== undefined) {
      e.dettaglio = dettaglio;
    }
    voce.eventi.push(e);
    voce.aggiornato_il = e.quando;
  }

  /* ------------------------------------------------------------ unità del campione e casi mirati */

  function bozzaVuota() {
    return {risposta: "", non_riesco: false, leggibilita: NON, note: ""};
  }

  function dopoVuoto() {
    return {corretta: NON, coerente_con_annotazioni: NON, rispondibile_da_immagine: NON, gravita: NON,
            tipo_difetto: "", note: "", risposta_dopo_soluzione: ""};
  }

  function voce(stato, unit, crea) {
    var v = stato.unita[unit.id];
    if (!v && crea) {
      v = {tipo: "unita", varianti: {}, bozza: {}, prima: null, soluzione_mostrata_il: null, dopo: {varianti: {}},
           tempo_ms: 0, eventi: [], aggiornato_il: ora()};
      unit.varianti.forEach(function (k) {
        v.varianti[k] = {record: unit.record_per_variante[k]};
        v.bozza[k] = bozzaVuota();
        v.dopo.varianti[k] = dopoVuoto();
      });
      if (unit.coppia) {
        v.dopo.coppia = {contratto_coppia: NON, gravita: NON, tipo_difetto: "", note: ""};
      }
      stato.unita[unit.id] = v;
    }
    return v || null;
  }

  function aggiornaBozza(stato, unit, variante, campo, valore) {
    var v = voce(stato, unit, true);
    if (v.soluzione_mostrata_il) {
      return ["la soluzione è già stata mostrata: il primo giudizio non si modifica più"];
    }
    if (!(variante in v.bozza) || !(campo in v.bozza[variante])) {
      return ["campo sconosciuto: " + variante + "/" + campo];
    }
    if (campo === "non_riesco") {
      valore = valore === true;
    } else if (campo === "leggibilita") {
      if (!haValore(ammessi("leggibilita"), valore)) {
        return ["valore non ammesso: " + valore];
      }
    } else {
      valore = String(valore);
    }
    v.bozza[variante][campo] = valore;
    v.aggiornato_il = ora();
    return [];
  }

  function erroriPrima(unit, bozza) {
    var errori = [];
    unit.varianti.forEach(function (k) {
      var b = bozza[k];
      var nome = k ? "variante " + k : "unità";
      if (unit.livello === "qa" && !b.non_riesco && !b.risposta.trim()) {
        errori.push(nome + ": scrivi la risposta oppure spunta \"non riesco a rispondere\"");
      }
      if (b.leggibilita === NON) {
        errori.push(nome + ": manca il giudizio iniziale di leggibilità e rispondibilità");
      }
    });
    return errori;
  }

  function copia(x) {
    return JSON.parse(JSON.stringify(x));
  }

  function salvaPrima(stato, unit) {
    var v = voce(stato, unit, true);
    if (v.soluzione_mostrata_il) {
      return ["la soluzione è già stata mostrata: il primo giudizio non si modifica più"];
    }
    var errori = erroriPrima(unit, v.bozza);
    if (errori.length) {
      return errori;
    }
    v.prima = {varianti: copia(v.bozza), salvata_il: ora(), salvataggi: v.prima ? v.prima.salvataggi + 1 : 1};
    evento(v, "primo_giudizio_salvato");
    return [];
  }

  function bozzaUgualeAPrima(v) {
    return !!(v && v.prima) && canonico(v.bozza) === canonico(v.prima.varianti);
  }

  function puoMostrare(stato, unit) {
    var v = voce(stato, unit, false);
    if (v && v.soluzione_mostrata_il) {
      return {ok: true, motivo: "già mostrata"};
    }
    if (!v || !v.prima) {
      return {ok: false, motivo: "prima salva il primo giudizio"};
    }
    if (!bozzaUgualeAPrima(v)) {
      return {ok: false, motivo: "hai modifiche non salvate al primo giudizio: salvale prima"};
    }
    return {ok: true, motivo: ""};
  }

  function mostraSoluzione(stato, unit) {
    var p = puoMostrare(stato, unit);
    if (!p.ok) {
      return [p.motivo];
    }
    var v = voce(stato, unit, true);
    if (!v.soluzione_mostrata_il) {
      v.soluzione_mostrata_il = ora();
      evento(v, "soluzione_mostrata");
    }
    return [];
  }

  function aggiornaDopo(stato, unit, parte, campo, valore) {
    var v = voce(stato, unit, false);
    if (!v || !v.soluzione_mostrata_il) {
      return ["il giudizio finale si compila dopo aver mostrato la soluzione"];
    }
    var obj = parte === "coppia" ? v.dopo.coppia : v.dopo.varianti[parte];
    if (!obj || !(campo in obj)) {
      return ["campo sconosciuto: " + parte + "/" + campo];
    }
    if (campo === "gravita") {
      if (!haValore(ammessi("gravita"), valore)) {
        return ["valore non ammesso: " + valore];
      }
    } else if (haValore(CAMPI_DOPO, campo) || campo === "contratto_coppia") {
      if (!haValore(ammessi("giudizio"), valore)) {
        return ["valore non ammesso: " + valore];
      }
    } else {
      valore = String(valore);
    }
    if (obj[campo] !== valore) {
      obj[campo] = valore;
      if (typeof obj[campo] === "string" && (campo === "note" || campo === "tipo_difetto"
                                             || campo === "risposta_dopo_soluzione")) {
        v.aggiornato_il = ora();  // testo libero: niente evento a ogni tasto
      } else {
        evento(v, "giudizio_finale", parte + "/" + campo + "=" + valore);
      }
    }
    return [];
  }

  function aggiungiTempo(stato, cosa, ms, decontam) {
    if (!(ms > 0)) {
      return;
    }
    var v = decontam ? voceDecontam(stato, cosa, true) : voce(stato, cosa, true);
    v.tempo_ms += Math.round(ms);
  }

  function valoriDi(v) {
    var out = [];
    if (!v) {
      return out;
    }
    if (v.prima) {
      Object.keys(v.prima.varianti).forEach(function (k) { out.push(v.prima.varianti[k].leggibilita); });
    }
    Object.keys(v.dopo.varianti).forEach(function (k) {
      CAMPI_DOPO.forEach(function (c) { out.push(v.dopo.varianti[k][c]); });
    });
    if (v.dopo.coppia) {
      out.push(v.dopo.coppia.contratto_coppia, v.dopo.coppia.gravita);
    }
    return out;
  }

  function mancantiDopo(unit, v) {
    var mancanti = [];
    unit.varianti.forEach(function (k) {
      var d = v.dopo.varianti[k];
      var nome = k ? "variante " + k : "unità";
      CAMPI_DOPO.forEach(function (c) {
        if (d[c] === NON) {
          mancanti.push(nome + ": " + c);
        }
      });
      if ((d.gravita === "minore" || d.gravita === "grave") && !d.tipo_difetto.trim()) {
        mancanti.push(nome + ": tipo_difetto");
      }
    });
    if (unit.coppia) {
      var c = v.dopo.coppia;
      if (c.contratto_coppia === NON) {
        mancanti.push("coppia: contratto_coppia");
      }
      if (c.gravita === NON) {
        mancanti.push("coppia: gravita");
      }
      if ((c.gravita === "minore" || c.gravita === "grave") && !c.tipo_difetto.trim()) {
        mancanti.push("coppia: tipo_difetto");
      }
    }
    return mancanti;
  }

  function bozzaToccata(v) {
    return Object.keys(v.bozza).some(function (k) {
      var b = v.bozza[k];
      return b.risposta !== "" || b.non_riesco || b.leggibilita !== NON || b.note !== "";
    });
  }

  /* fase: non_iniziato, bozza, primo_giudizio (salvato, soluzione nascosta), soluzione_vista, completo */
  function statoUnita(unit, v) {
    var valori = valoriDi(v);
    var out = {fase: "non_iniziato", mancanti: [], incerto: haValore(valori, "incerto"),
               non_giudicabile: haValore(valori, "non_giudicabile"), non_verificabile: haValore(valori, "non_verificabile")};
    if (!v) {
      return out;
    }
    if (!v.prima) {
      out.fase = bozzaToccata(v) ? "bozza" : "non_iniziato";
      out.mancanti = erroriPrima(unit, v.bozza);
      return out;
    }
    if (!v.soluzione_mostrata_il) {
      out.fase = "primo_giudizio";
      out.mancanti = ["soluzione non ancora mostrata"];
      return out;
    }
    out.mancanti = mancantiDopo(unit, v);
    out.fase = out.mancanti.length ? "soluzione_vista" : "completo";
    return out;
  }

  /* ------------------------------------------------------------ decontaminazione */

  function voceDecontam(stato, d, crea) {
    var v = stato.decontaminazione[d.id];
    if (!v && crea) {
      v = {tipo: "decontaminazione", record: d.record, corrispondenze: {}, esito: NON, decisiva: "", note: "",
           tempo_ms: 0, eventi: [], aggiornato_il: ora()};
      d.elenco.forEach(function (x) { v.corrispondenze[x.id] = {esame: "non_esaminata", note: ""}; });
      stato.decontaminazione[d.id] = v;
    }
    return v || null;
  }

  function materialeMancante(d) {
    var out = [];
    if (d.materiale !== "completo") {
      out.push("materiale del record: " + d.materiale);
    }
    d.elenco.forEach(function (x) {
      if (!x.immagine) {
        out.push(x.id + ": immagine del pannello assente");
      }
      if (x.problemi) {
        out.push(x.id + ": " + x.problemi);
      }
    });
    return out;
  }

  function controlloDecontam(d, v) {
    var esami = d.elenco.map(function (x) { return v ? v.corrispondenze[x.id].esame : "non_esaminata"; });
    var esaminate = esami.filter(function (e) { return e !== "non_esaminata"; }).length;
    var out = {totale: d.elenco.length, esaminate: esaminate, errori: [], fase: "non_iniziato",
               incerto: !!v && (v.esito === "incerto" || haValore(esami, "incerto")),
               non_giudicabile: !!v && (v.esito === "non_giudicabile" || haValore(esami, "non_giudicabile"))};
    if (!v) {
      return out;
    }
    var esito = v.esito;
    if (esito === "no") {
      var nonDiverse = d.elenco.filter(function (x) { return v.corrispondenze[x.id].esame !== "contenuto_diverso"; });
      if (nonDiverse.length) {
        out.errori.push("per dichiarare \"no\" tutte le corrispondenze devono essere esaminate come contenuto " +
                        "diverso; restano " + nonDiverse.length + " (" + nonDiverse.map(function (x) {
                          return x.id; }).slice(0, 5).join(", ") + ")");
      }
      materialeMancante(d).forEach(function (m) { out.errori.push("materiale mancante: " + m); });
    }
    if (esito === "si") {
      if (!v.decisiva) {
        out.errori.push("indica la corrispondenza decisiva");
      } else if (!v.corrispondenze[v.decisiva] || v.corrispondenze[v.decisiva].esame !== "stesso_contenuto") {
        out.errori.push("la corrispondenza decisiva deve essere esaminata come stesso contenuto");
      }
    }
    if (esito !== "si" && v.decisiva) {
      out.errori.push("una corrispondenza decisiva si indica solo con l'esito \"sì\"");
    }
    var toccata = esaminate > 0 || esito !== NON || v.note !== "" || d.elenco.some(function (x) {
      return v.corrispondenze[x.id].note !== ""; });
    if (esito !== NON && !out.errori.length) {
      out.fase = "completo";
    } else if (toccata) {
      out.fase = "in_corso";
    }
    return out;
  }

  function puoEscludere(d, v) {
    var prova = copia(v || {corrispondenze: {}, esito: NON, decisiva: "", note: ""});
    d.elenco.forEach(function (x) {
      if (!prova.corrispondenze[x.id]) {
        prova.corrispondenze[x.id] = {esame: "non_esaminata", note: ""};
      }
    });
    prova.esito = "no";
    prova.decisiva = "";
    return controlloDecontam(d, prova).errori;
  }

  /* Restituisce {errori, avvisi}. Se il cambio rende incoerente un "no" già dato, il "no" si annulla (torna non
     compilato): un'esclusione della corrispondenza col pannello non resta mai con candidati non esaminati. */
  function aggiornaEsame(stato, d, idCorr, campo, valore) {
    var v = voceDecontam(stato, d, true);
    var c = v.corrispondenze[idCorr];
    if (!c || !(campo in c)) {
      return {errori: ["corrispondenza o campo sconosciuti: " + idCorr + "/" + campo], avvisi: []};
    }
    if (campo === "esame" && !haValore(ammessi("esame"), valore)) {
      return {errori: ["valore non ammesso: " + valore], avvisi: []};
    }
    c[campo] = campo === "esame" ? valore : String(valore);
    if (campo === "esame") {
      evento(v, "corrispondenza", idCorr + "=" + valore);
    } else {
      v.aggiornato_il = ora();
    }
    var avvisi = [];
    if (v.esito === "no" && puoEscludere(d, v).length) {
      v.esito = NON;
      evento(v, "esito_annullato", "il no non vale più dopo il cambio di " + idCorr);
      avvisi.push("L'esito \"no\" è stato annullato: con questo cambio non tutte le corrispondenze sono più " +
                  "contenuto diverso.");
    }
    return {errori: [], avvisi: avvisi};
  }

  function impostaEsitoDecontam(stato, d, campo, valore) {
    var v = voceDecontam(stato, d, true);
    if (campo === "esito") {
      if (!haValore(ammessi("esito_decontaminazione"), valore)) {
        return ["valore non ammesso: " + valore];
      }
      if (valore === "no") {
        var errori = puoEscludere(d, v);
        if (errori.length) {
          return errori;
        }
      }
      v.esito = valore;
      if (valore !== "si") {
        v.decisiva = "";
      }
    } else if (campo === "decisiva") {
      if (valore && !v.corrispondenze[valore]) {
        return ["corrispondenza sconosciuta: " + valore];
      }
      v.decisiva = valore;
    } else if (campo === "note") {
      v.note = String(valore);
      v.aggiornato_il = ora();
      return [];
    } else {
      return ["campo sconosciuto: " + campo];
    }
    evento(v, "esito", campo + "=" + valore);
    return [];
  }

  /* ------------------------------------------------------------ riepilogo */

  function riepilogo(pkg, stato) {
    var out = {};
    ["campione", "mirati"].forEach(function (sezione) {
      var tot = {totale: 0, gruppi: {}};
      pkg[sezione].forEach(function (u) {
        var s = statoUnita(u, stato.unita[u.id]);
        var g = tot.gruppi[u.gruppo] = tot.gruppi[u.gruppo] || {totale: 0};
        [tot, g].forEach(function (c) {
          c.totale += 1;
          c[s.fase] = (c[s.fase] || 0) + 1;
          ["incerto", "non_giudicabile", "non_verificabile"].forEach(function (f) {
            if (s[f]) {
              c[f] = (c[f] || 0) + 1;
            }
          });
        });
      });
      out[sezione] = tot;
    });
    var d = {totale: 0};
    pkg.decontaminazione.forEach(function (x) {
      var s = controlloDecontam(x, stato.decontaminazione[x.id]);
      d.totale += 1;
      d[s.fase] = (d[s.fase] || 0) + 1;
      if (s.incerto) {
        d.incerto = (d.incerto || 0) + 1;
      }
      if (s.non_giudicabile) {
        d.non_giudicabile = (d.non_giudicabile || 0) + 1;
      }
      d.corrispondenze_esaminate = (d.corrispondenze_esaminate || 0) + s.esaminate;
      d.corrispondenze = (d.corrispondenze || 0) + s.totale;
    });
    out.decontaminazione = d;
    return out;
  }

  /* ------------------------------------------------------------ esportazione */

  function canonico(x) {
    if (x === null || typeof x !== "object") {
      return JSON.stringify(x);
    }
    if (Array.isArray(x)) {
      return "[" + x.map(canonico).join(",") + "]";
    }
    return "{" + Object.keys(x).sort().map(function (k) {
      return JSON.stringify(k) + ":" + canonico(x[k]);
    }).join(",") + "}";
  }

  function oggettoEsportato(pkg, stato) {
    return {schema: SCHEMA_STATO, pacchetto: {id: pkg.id, versione: pkg.versione, impronta: pkg.impronta},
            revisore: stato.revisore, creato_il: stato.creato_il, aggiornato_il: stato.aggiornato_il,
            esportato_il: ora(), posizione: stato.posizione, unita: stato.unita,
            decontaminazione: stato.decontaminazione, riepilogo: riepilogo(pkg, stato)};
  }

  function esportaJSON(pkg, stato) {
    return JSON.stringify(oggettoEsportato(pkg, stato), null, 1) + "\n";
  }

  function cella(valore) {
    return String(valore === undefined || valore === null ? "" : valore)
      .replace(/\\/g, "\\\\").replace(/\r\n|\r|\n/g, "\\n").replace(/\t/g, "\\t");
  }

  function leggiCella(testo) {
    return testo.replace(/\\(\\|n|t)/g, function (_, c) { return c === "n" ? "\n" : c === "t" ? "\t" : "\\"; });
  }

  function rigaTsv(valori) {
    return valori.map(cella).join("\t");
  }

  function tsvEsiti(pkg, stato) {
    var idx = indice(pkg);
    var righe = [TSV_ESITI.join("\t")];
    pkg.righe_esiti.forEach(function (r) {
      var unit = idx.unita[r.unita];
      var v = stato.unita[r.unita];
      var tempo = v ? String(Math.round(v.tempo_ms / 1000)) : "";
      var row;
      if (r.variante === "coppia") {
        var c = v && v.soluzione_mostrata_il ? v.dopo.coppia : null;
        row = [r.unita, r.variante, r.record, "non_applicabile", "non_applicabile", "non_applicabile",
               "non_applicabile", "non_applicabile", c ? c.contratto_coppia : NON, c ? c.gravita : NON,
               c ? c.tipo_difetto : "", tempo, c ? c.note : ""];
      } else {
        var p = v && v.prima ? v.prima.varianti[r.variante] : null;
        var d = v && v.soluzione_mostrata_il ? v.dopo.varianti[r.variante] : null;
        var risposta = !p ? NON : p.non_riesco ? "non_riesco_a_rispondere" : p.risposta;
        if (!p && unit.livello === "scena") {
          risposta = NON;
        } else if (p && unit.livello === "scena" && !p.non_riesco && !p.risposta.trim()) {
          risposta = "nessuna_risposta_caso_sulla_scena";
        }
        row = [r.unita, r.variante, r.record, risposta, d ? d.corretta : NON, d ? d.coerente_con_annotazioni : NON,
               d ? d.rispondibile_da_immagine : NON, "non_applicabile", "non_applicabile", d ? d.gravita : NON,
               d ? d.tipo_difetto : "", tempo, d ? d.note : ""];
      }
      righe.push(rigaTsv(row));
    });
    return righe.join("\n") + "\n";
  }

  function tsvDecontaminazione(pkg, stato) {
    var idx = indice(pkg);
    var righe = [TSV_DECONTAMINAZIONE.join("\t")];
    pkg.righe_esiti_decontaminazione.forEach(function (r) {
      var d = idx.decontaminazione[r.unita].d;
      var v = stato.decontaminazione[r.unita];
      var s = controlloDecontam(d, v);
      var guardate = d.elenco.map(function (x) {
        return x.id + "=" + (v ? v.corrispondenze[x.id].esame : "non_esaminata");
      }).join("; ");
      var esito = v ? v.esito : NON;
      if (s.errori.length && esito !== NON) {
        esito = "incompleto_" + esito;  // per esempio "sì" senza la corrispondenza decisiva
      }
      var note = v ? v.note : "";
      if (v) {
        d.elenco.forEach(function (x) {
          if (v.corrispondenze[x.id].note) {
            note += (note ? "\n" : "") + x.id + ": " + v.corrispondenze[x.id].note;
          }
        });
      }
      righe.push(rigaTsv([r.unita, r.record, esito, s.esaminate + "/" + s.totale + ": " + guardate,
                          v && v.esito === "si" ? v.decisiva : "", note]));
    });
    return righe.join("\n") + "\n";
  }

  /* ------------------------------------------------------------ importazione */

  function tipoOk(x, tipo) {
    return tipo === "array" ? Array.isArray(x) : typeof x === tipo && x !== null && !Array.isArray(x);
  }

  function chiaviUguali(obj, attese) {
    return tipoOk(obj, "object") && canonico(Object.keys(obj).sort()) === canonico(attese.slice().sort());
  }

  function controllaVoceUnita(unit, v, errori) {
    var dove = unit.id + ": ";
    var tutte = ["tipo", "varianti", "bozza", "prima", "soluzione_mostrata_il", "dopo", "tempo_ms", "eventi",
                 "aggiornato_il"];
    if (!chiaviUguali(v, tutte) || v.tipo !== "unita") {
      errori.push(dove + "struttura non riconosciuta");
      return;
    }
    if (!chiaviUguali(v.varianti, unit.varianti) || !chiaviUguali(v.bozza, unit.varianti)
        || !tipoOk(v.dopo, "object") || !chiaviUguali(v.dopo.varianti, unit.varianti)) {
      errori.push(dove + "varianti diverse da quelle dell'unità");
      return;
    }
    unit.varianti.forEach(function (k) {
      if (!v.varianti[k] || v.varianti[k].record !== unit.record_per_variante[k]) {
        errori.push(dove + "record della variante " + (k || "unica") + " diverso da quello del pacchetto");
      }
      var b = v.bozza[k];
      if (!chiaviUguali(b, ["risposta", "non_riesco", "leggibilita", "note"]) || typeof b.risposta !== "string"
          || typeof b.note !== "string" || typeof b.non_riesco !== "boolean"
          || !haValore(ammessi("leggibilita"), b.leggibilita)) {
        errori.push(dove + "bozza non valida");
      }
      var d = v.dopo.varianti[k];
      if (!chiaviUguali(d, ["corretta", "coerente_con_annotazioni", "rispondibile_da_immagine", "gravita",
                            "tipo_difetto", "note", "risposta_dopo_soluzione"])
          || !haValore(ammessi("giudizio"), d.corretta) || !haValore(ammessi("giudizio"), d.coerente_con_annotazioni)
          || !haValore(ammessi("giudizio"), d.rispondibile_da_immagine) || !haValore(ammessi("gravita"), d.gravita)
          || typeof d.tipo_difetto !== "string" || typeof d.note !== "string"
          || typeof d.risposta_dopo_soluzione !== "string") {
        errori.push(dove + "giudizio finale non valido");
      }
    });
    if (unit.coppia) {
      var c = v.dopo.coppia;
      if (!chiaviUguali(c, ["contratto_coppia", "gravita", "tipo_difetto", "note"])
          || !haValore(ammessi("giudizio"), c.contratto_coppia) || !haValore(ammessi("gravita"), c.gravita)
          || typeof c.tipo_difetto !== "string" || typeof c.note !== "string") {
        errori.push(dove + "giudizio della coppia non valido");
      }
      if (!chiaviUguali(v.dopo, ["varianti", "coppia"])) {
        errori.push(dove + "giudizio finale senza la parte della coppia");
      }
    } else if (!chiaviUguali(v.dopo, ["varianti"])) {
      errori.push(dove + "parte della coppia in un'unità che non è una coppia");
    }
    if (v.prima !== null) {
      if (!chiaviUguali(v.prima, ["varianti", "salvata_il", "salvataggi"]) || !chiaviUguali(v.prima.varianti, unit.varianti)
          || typeof v.prima.salvata_il !== "string" || unit.varianti.some(function (k) {
            var p = v.prima.varianti[k];
            return !chiaviUguali(p, ["risposta", "non_riesco", "leggibilita", "note"]) || typeof p.risposta !== "string"
              || typeof p.note !== "string" || typeof p.non_riesco !== "boolean"
              || !haValore(ammessi("leggibilita"), p.leggibilita);
          })) {
        errori.push(dove + "primo giudizio non valido");
      } else if (erroriPrima(unit, v.prima.varianti).length) {
        errori.push(dove + "primo giudizio incompleto");
      }
    }
    if (v.soluzione_mostrata_il !== null && (typeof v.soluzione_mostrata_il !== "string" || v.prima === null)) {
      errori.push(dove + "soluzione mostrata senza un primo giudizio salvato");
    }
    if (typeof v.tempo_ms !== "number" || !tipoOk(v.eventi, "array")) {
      errori.push(dove + "tempo o eventi non validi");
    }
  }

  function controllaVoceDecontam(d, v, errori) {
    var dove = d.id + ": ";
    if (!chiaviUguali(v, ["tipo", "record", "corrispondenze", "esito", "decisiva", "note", "tempo_ms", "eventi",
                          "aggiornato_il"]) || v.tipo !== "decontaminazione") {
      errori.push(dove + "struttura non riconosciuta");
      return;
    }
    if (v.record !== d.record) {
      errori.push(dove + "record diverso da quello del pacchetto");
    }
    if (!chiaviUguali(v.corrispondenze, d.elenco.map(function (x) { return x.id; }))) {
      errori.push(dove + "corrispondenze diverse da quelle del pacchetto");
      return;
    }
    d.elenco.forEach(function (x) {
      var c = v.corrispondenze[x.id];
      if (!chiaviUguali(c, ["esame", "note"]) || !haValore(ammessi("esame"), c.esame) || typeof c.note !== "string") {
        errori.push(dove + x.id + " non valida");
      }
    });
    if (!haValore(ammessi("esito_decontaminazione"), v.esito) || typeof v.decisiva !== "string"
        || typeof v.note !== "string" || typeof v.tempo_ms !== "number" || !tipoOk(v.eventi, "array")) {
      errori.push(dove + "esito, note o tempo non validi");
    } else if (v.esito === "no" && puoEscludere(d, v).length) {
      errori.push(dove + "esito \"no\" con corrispondenze non esaminate come contenuto diverso o materiale mancante");
    } else if (v.decisiva && !v.corrispondenze[v.decisiva]) {
      errori.push(dove + "corrispondenza decisiva sconosciuta");
    }
  }

  function controllaImport(pkg, testo) {
    var errori = [];
    var dati;
    try {
      dati = JSON.parse(testo);
    } catch (e) {
      return {errori: ["il file non è un JSON valido: " + e.message], stato: null};
    }
    if (!tipoOk(dati, "object") || dati.schema !== SCHEMA_STATO) {
      return {errori: ["schema del file " + (dati && dati.schema) + ", atteso " + SCHEMA_STATO], stato: null};
    }
    var p = dati.pacchetto || {};
    if (p.id !== pkg.id || p.versione !== pkg.versione || p.impronta !== pkg.impronta) {
      return {errori: ["il file appartiene a un altro pacchetto o a un'altra versione: " + p.id + " v" + p.versione +
                       ", impronta " + String(p.impronta).slice(0, 12) + " (questo pacchetto: " + pkg.id + " v" +
                       pkg.versione + ", " + pkg.impronta.slice(0, 12) + ")"], stato: null};
    }
    if (!tipoOk(dati.unita, "object") || !tipoOk(dati.decontaminazione, "object")) {
      return {errori: ["mancano le sezioni unita o decontaminazione"], stato: null};
    }
    var idx = indice(pkg);
    Object.keys(dati.unita).forEach(function (id) {
      if (!idx.unita[id]) {
        errori.push(id + ": unità sconosciuta in questo pacchetto");
      } else {
        controllaVoceUnita(idx.unita[id], dati.unita[id], errori);
      }
    });
    Object.keys(dati.decontaminazione).forEach(function (id) {
      if (!idx.decontaminazione[id]) {
        errori.push(id + ": candidato sconosciuto in questo pacchetto");
      } else {
        controllaVoceDecontam(idx.decontaminazione[id].d, dati.decontaminazione[id], errori);
      }
    });
    if (errori.length) {
      return {errori: errori, stato: null};
    }
    var stato = {schema: SCHEMA_STATO, pacchetto: {id: pkg.id, versione: pkg.versione, impronta: pkg.impronta},
                 revisore: typeof dati.revisore === "string" ? dati.revisore : "",
                 creato_il: dati.creato_il || ora(), aggiornato_il: dati.aggiornato_il || ora(),
                 unita: dati.unita, decontaminazione: dati.decontaminazione,
                 posizione: typeof dati.posizione === "string" ? dati.posizione : null};
    return {errori: [], stato: stato};
  }

  function senzaContatori(v) {
    var c = copia(v);
    delete c.tempo_ms;
    delete c.eventi;
    delete c.aggiornato_il;
    return canonico(c);
  }

  /* Confronto fra lo stato corrente e uno importato: le voci presenti in uno solo si uniscono, quelle uguali (a meno
     di tempo, eventi e ora dell'ultima modifica) restano, quelle diverse sono conflitti e non si uniscono. */
  function confronta(corrente, importato) {
    var unito = copia(corrente);
    var out = {conflitti: [], nuove: 0, uguali: 0, solo_correnti: 0, unito: unito};
    ["unita", "decontaminazione"].forEach(function (sezione) {
      var a = corrente[sezione], b = importato[sezione];
      Object.keys(b).forEach(function (id) {
        if (!a[id]) {
          unito[sezione][id] = copia(b[id]);
          out.nuove += 1;
        } else if (senzaContatori(a[id]) === senzaContatori(b[id])) {
          out.uguali += 1;
          var u = unito[sezione][id];
          u.tempo_ms = Math.max(a[id].tempo_ms, b[id].tempo_ms);
          if (b[id].eventi.length > a[id].eventi.length) {
            u.eventi = copia(b[id].eventi);
          }
        } else {
          out.conflitti.push({id: id, sezione: sezione, corrente: a[id].aggiornato_il, file: b[id].aggiornato_il});
        }
      });
      Object.keys(a).forEach(function (id) {
        if (!b[id]) {
          out.solo_correnti += 1;
        }
      });
    });
    if (!unito.revisore && importato.revisore) {
      unito.revisore = importato.revisore;
    }
    return out;
  }

  root.Logica = {
    SCHEMA_STATO: SCHEMA_STATO, NON: NON, VALORI: VALORI, AIUTI: AIUTI, TIPI_DIFETTO: TIPI_DIFETTO,
    TSV_ESITI: TSV_ESITI, TSV_DECONTAMINAZIONE: TSV_DECONTAMINAZIONE, CAMPI_DOPO: CAMPI_DOPO,
    ammessi: ammessi, ora: ora, indice: indice, nuovoStato: nuovoStato, voce: voce, aggiornaBozza: aggiornaBozza,
    erroriPrima: erroriPrima, salvaPrima: salvaPrima, bozzaUgualeAPrima: bozzaUgualeAPrima, puoMostrare: puoMostrare,
    mostraSoluzione: mostraSoluzione, aggiornaDopo: aggiornaDopo, aggiungiTempo: aggiungiTempo, statoUnita: statoUnita,
    voceDecontam: voceDecontam, materialeMancante: materialeMancante, controlloDecontam: controlloDecontam,
    puoEscludere: puoEscludere, aggiornaEsame: aggiornaEsame, impostaEsitoDecontam: impostaEsitoDecontam,
    riepilogo: riepilogo, canonico: canonico, oggettoEsportato: oggettoEsportato, esportaJSON: esportaJSON,
    cella: cella, leggiCella: leggiCella, tsvEsiti: tsvEsiti, tsvDecontaminazione: tsvDecontaminazione,
    controllaImport: controllaImport, confronta: confronta, copia: copia
  };
})(window);
