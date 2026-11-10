/* Logica della revisione visiva offline di MAVIS (C9, campione-1): stato, primo e secondo passaggio, collegamenti fra
   schede ed esposizioni dei record, esportazione, importazione e unione degli stati. Nessun accesso al DOM né alla
   rete: la usano index.html (app.js) e verifica.html (test.js). Deriva dalla logica di revisioni/synth-v4-campione-2,
   senza coppie, decontaminazione e scene.

   Esposizioni. Mostrare la soluzione di una scheda espone il suo record e il suo gruppo: target candidato, annotazioni
   e motivi automatici. Le schede che mostrano lo stesso record o lo stesso gruppo sono collegate, quindi:
   - la soluzione si mostra solo quando tutte le schede collegate hanno un primo giudizio salvato;
   - al momento della rivelazione il primo giudizio di queste schede resta fissato (prima_fissata);
   - ogni esposizione si registra in stato.esposizioni e non si cancella più;
   - una scheda esposta prima di avere un primo giudizio salvato è segnata (esposta_prima): il suo primo giudizio non è
     indipendente;
   - unendo due stati (importazione, sostituzione, ripresa nel browser) esposizioni e primi giudizi fissati restano; le
     versioni scartate vanno in stato.archivio.
   In questo pacchetto le schede hanno record e gruppi distinti, quindi nessuna è collegata; la protezione resta. */
(function (root) {
  "use strict";

  var SCHEMA_STATO = "mavis-c9-revisione-avanzamento/1";
  var NON = "non_compilato";
  var NA = "non_applicabile";
  var QUATTRO = [
    [NON, "— non ancora compilato —"],
    ["si", "sì"],
    ["no", "no"],
    ["incerto", "incerto: ho guardato ma non so decidere"],
    ["non_giudicabile", "non giudicabile: il caso non permette un giudizio"]
  ];

  var VALORI = {
    leggibilita: [
      [NON, "— non ancora compilato —"],
      ["si", "sì: ciò che serve è visibile, leggibile e univoco"],
      ["no", "no: manca, è coperto, tagliato, illeggibile o ambiguo"],
      ["incerto", "incerto: ho guardato ma non so decidere"],
      ["non_giudicabile", "non giudicabile: il caso non permette un giudizio"]
    ],
    sufficienza: [
      [NON, "— non ancora compilato —"],
      ["si", "sì: immagine e domanda bastano per rispondere"],
      ["no", "no: manca un'informazione necessaria"],
      ["incerto", "incerto: ho guardato ma non so decidere"],
      ["non_giudicabile", "non giudicabile: il caso non permette un giudizio"]
    ],
    corretta_pixel: QUATTRO,
    rispondibile: QUATTRO,
    scrittura_visibile: QUATTRO,
    prudenza_eccessiva: QUATTRO,
    giudizio_finale: [
      [NON, "— non ancora compilato —"],
      ["nessun_problema", "nessun problema"],
      ["errore_grave", "errore grave: target o annotazione sbagliati, domanda non rispondibile"],
      ["problema_minore", "problema minore di formato o metadato: risposta e lettura non cambiano"],
      ["ambiguita", "ambiguità: più letture o più risposte plausibili"],
      ["non_giudicabile", "caso non giudicabile"]
    ]
  };

  var AIUTI = {
    risposta: "La tua risposta, scritta prima di vedere la soluzione. Per un valore scritto nella figura riporta la " +
              "scrittura che vedi, senza semplificarla. Per una richiesta diagnostica elenca tutte le risposte vere.",
    non_determinabile: "Spunta se dall'immagine e dalla domanda non riesci a rispondere: vale come risposta.",
    leggibilita: "Ciò che serve per rispondere è visibile, leggibile alla risoluzione dell'originale e univoco?",
    sufficienza: "Immagine e domanda bastano, senza informazioni esterne?",
    note_prima: "Facoltative: che cosa ti ha fatto esitare, dove guardare.",
    corretta_pixel: "Il target candidato è giusto per ciò che si vede nei pixel?",
    corretta_pixel_diagnostica: "Tipo dichiarato e misure delle annotazioni corrispondono ai pixel?",
    rispondibile: "Si risponde con la sola immagine e la domanda?",
    rispondibile_diagnostica: "Con la sola immagine si decide quali classi sono vere?",
    scrittura_visibile: "La scrittura visibile coincide con il visualizzato previsto, nella stessa forma (una radice " +
                        "resta una radice, un decimale resta un decimale)?",
    equivalenza: "Facoltativa: equivalenze matematiche, per esempio √841 = 29. Restano separate e non correggono la " +
                 "trascrizione.",
    prudenza_eccessiva: "Il filtro è stato troppo prudente? Sì se dalle misure stampate e dalla figura una sola " +
                        "risposta dell'insieme sarebbe dimostrabile.",
    risposte_vere: "Facoltativa: le classi che ritieni vere dopo aver visto le annotazioni (non indipendente).",
    giudizio_finale: "Grave = errore; minore = formato o metadato, senza cambiare risposta e lettura; ambiguità = più " +
                     "letture plausibili; non giudicabile = il caso non permette un giudizio.",
    tipo_difetto: "Obbligatorio con errore grave, problema minore o ambiguità: scegli un tipo o scrivine uno.",
    note_dopo: "Note del giudizio finale; più righe ammesse.",
    risposta_dopo_soluzione: "Facoltativa: la tua risposta dopo la soluzione. Non è indipendente e resta separata.",
    collegate: "Queste schede mostrano lo stesso record o lo stesso gruppo: la soluzione rivela anche le loro " +
               "risposte. Prima di mostrarla salva il loro primo giudizio, che da quel momento resta fissato."
  };

  var TIPI_DIFETTO = [
    "target_errato", "annotazione_errata", "valore_illeggibile", "scrittura_diversa_dal_previsto",
    "figura_diversa_dalle_annotazioni", "elemento_nascosto_o_tagliato", "domanda_ambigua", "piu_risposte_vere",
    "filtro_troppo_prudente", "filtro_troppo_permissivo", "metadato_errato", "formato", "altro"
  ];

  var SCELTE_DOPO = ["corretta_pixel", "rispondibile", "scrittura_visibile", "prudenza_eccessiva", "giudizio_finale"];
  var TESTI_DOPO = ["tipo_difetto", "equivalenza", "risposte_vere", "note", "risposta_dopo_soluzione"];
  var SERVE_DIFETTO = ["errore_grave", "problema_minore", "ambiguita"];
  var TSV_ESITI = ["scheda", "record", "controllo_noto", "indipendente", "risposta_revisore", "non_determinabile",
                   "leggibilita", "sufficienza", "note_prima", "soluzione_mostrata_il", "corretta_pixel",
                   "rispondibile", "scrittura_visibile", "prudenza_eccessiva", "giudizio_finale", "tipo_difetto",
                   "equivalenza", "risposte_vere", "risposta_dopo_soluzione", "note_dopo", "tempo_s"];

  function ammessi(nome) {
    return VALORI[nome].map(function (x) { return x[0]; });
  }

  function ora() {
    return new Date().toISOString();
  }

  function copia(x) {
    return JSON.parse(JSON.stringify(x));
  }

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

  function aggiungiUnico(lista, x) {
    if (lista.indexOf(x) < 0) {
      lista.push(x);
    }
  }

  /* ------------------------------------------------------------ contesto e collegamenti */

  var cache = {pkg: null, sol: null, ctx: null};

  /* Contesto: indice delle schede, modulo del secondo passaggio di ognuna (dalle soluzioni, mai mostrato prima) e
     collegamenti. colpiteDi[record o gruppo]: le schede di cui l'esposizione rivela la risposta; collegati[scheda]: le
     altre schede colpite dalla sua soluzione. */
  function contesto(pkg, sol) {
    if (cache.ctx && cache.pkg === pkg && cache.sol === sol) {
      return cache.ctx;
    }
    var ctx = {pkg: pkg, sol: sol, idx: {}, ordine: [], colpiteDi: {}, collegati: {}};
    pkg.schede.forEach(function (s) {
      ctx.idx[s.id] = s;
      ctx.ordine.push(s.id);
      ["record:" + s.record, "gruppo:" + s.gruppo].forEach(function (k) {
        (ctx.colpiteDi[k] = ctx.colpiteDi[k] || []).push(s.id);
      });
    });
    pkg.schede.forEach(function (s) {
      var out = [];
      [["record:" + s.record, "stesso_record"], ["gruppo:" + s.gruppo, "stesso_gruppo"]].forEach(function (c) {
        ctx.colpiteDi[c[0]].forEach(function (id) {
          if (id !== s.id && !out.some(function (x) { return x.id === id; })) {
            out.push({id: id, collegamento: c[1]});
          }
        });
      });
      ctx.collegati[s.id] = out;
    });
    cache = {pkg: pkg, sol: sol, ctx: ctx};
    return ctx;
  }

  /* Campi del secondo passaggio che riguardano la scheda: la scrittura visibile solo per le letture di valori, la
     prudenza del filtro solo per le richieste diagnostiche. */
  function modulo(ctx, id) {
    var s = ctx.idx[id], sol = ctx.sol[id] || {};
    var lettura = sol.categoria === "lettura_valore" || sol.categoria === "controllo_noto";
    return {lettura: lettura, diagnostica: s.richiesta === "diagnostica",
            scelte: SCELTE_DOPO.filter(function (c) {
              return (c !== "scrittura_visibile" || lettura) && (c !== "prudenza_eccessiva" || s.richiesta === "diagnostica");
            })};
  }

  /* ------------------------------------------------------------ stato */

  function nuovoStato(pkg) {
    var t = ora();
    return {schema: SCHEMA_STATO, pacchetto: {id: pkg.id, versione: pkg.versione, impronta: pkg.impronta},
            revisore: "", creato_il: t, aggiornato_il: t, schede: {}, posizione: null, esposizioni: [], archivio: []};
  }

  function bozzaVuota() {
    return {risposta: "", non_determinabile: false, leggibilita: NON, sufficienza: NON, note: ""};
  }

  function dopoVuoto() {
    return {corretta_pixel: NON, rispondibile: NON, scrittura_visibile: NON, prudenza_eccessiva: NON,
            giudizio_finale: NON, tipo_difetto: "", equivalenza: "", risposte_vere: "", note: "",
            risposta_dopo_soluzione: ""};
  }

  function voce(stato, s, crea) {
    var v = stato.schede[s.id];
    if (!v && crea) {
      v = {bozza: bozzaVuota(), prima: null, prima_fissata: null, esposta_prima: null, soluzione_mostrata_il: null,
           dopo: dopoVuoto(), tempo_ms: 0, eventi: [], aggiornato_il: ora()};
      stato.schede[s.id] = v;
    }
    return v || null;
  }

  function evento(v, azione, dettaglio) {
    var e = {quando: ora(), azione: azione};
    if (dettaglio !== undefined) {
      e.dettaglio = dettaglio;
    }
    v.eventi.push(e);
    v.aggiornato_il = e.quando;
  }

  function archivia(stato, id, origine, motivo, contenuto) {
    var a = {quando: ora(), id: id, origine: origine, motivo: motivo, contenuto: copia(contenuto)};
    var k = id + "|" + canonico(a.contenuto);
    if (stato.archivio.some(function (x) { return x.id + "|" + canonico(x.contenuto) === k; })) {
      return false;
    }
    stato.archivio.push(a);
    return true;
  }

  function chiaveEsposizione(e) {
    return e.chiave + "|" + e.tramite + "|" + e.quando;
  }

  function aggiungiEsposizione(stato, e) {
    var k = chiaveEsposizione(e);
    if (stato.esposizioni.some(function (x) { return chiaveEsposizione(x) === k; })) {
      return false;
    }
    stato.esposizioni.push({chiave: e.chiave, quando: e.quando, tramite: e.tramite});
    return true;
  }

  /* Esposizioni registrate che riguardano ogni scheda, la più antica per prima. */
  function primaEsposizione(ctx, stato, id) {
    var s = ctx.idx[id], best = null;
    stato.esposizioni.forEach(function (e) {
      if ((e.chiave === "record:" + s.record || e.chiave === "gruppo:" + s.gruppo) && (!best || e.quando < best.quando)) {
        best = e;
      }
    });
    return best;
  }

  /* Segna le schede esposte prima del loro primo giudizio salvato (non indipendenti). Restituisce le schede appena
     riclassificate. Il primo giudizio salvato prima di ogni esposizione resta indipendente. */
  function applicaEsposizioni(ctx, stato) {
    var nuove = [];
    ctx.ordine.forEach(function (id) {
      var e = primaEsposizione(ctx, stato, id);
      var v = stato.schede[id];
      if (!e || (v && v.prima && v.prima.salvata_il <= e.quando)) {
        return;
      }
      v = voce(stato, ctx.idx[id], true);
      if (!v.esposta_prima || e.quando < v.esposta_prima.quando) {
        v.esposta_prima = {quando: e.quando, tramite: e.tramite, chiave: e.chiave};
        evento(v, "esposta_prima_del_primo_giudizio", e.chiave + " tramite " + e.tramite);
        nuove.push(id);
      }
    });
    return nuove;
  }

  /* ------------------------------------------------------------ primo passaggio */

  function erroreFissato(v) {
    return "il primo giudizio è fissato dal " + v.prima_fissata.quando + " (" + v.prima_fissata.motivo + "): non si " +
      "modifica più; i ripensamenti vanno nel giudizio finale";
  }

  function aggiornaBozza(stato, s, campo, valore) {
    var v = voce(stato, s, true);
    if (v.prima_fissata) {
      return [erroreFissato(v)];
    }
    if (!(campo in v.bozza)) {
      return ["campo sconosciuto: " + campo];
    }
    if (campo === "non_determinabile") {
      valore = valore === true;
    } else if (campo === "leggibilita" || campo === "sufficienza") {
      if (ammessi(campo).indexOf(valore) < 0) {
        return ["valore non ammesso: " + valore];
      }
    } else {
      valore = String(valore);
    }
    v.bozza[campo] = valore;
    v.aggiornato_il = ora();
    return [];
  }

  function erroriPrima(b) {
    var e = [];
    if (!b.non_determinabile && !b.risposta.trim()) {
      e.push("scrivi la risposta oppure spunta \"non determinabile\"");
    }
    if (b.leggibilita === NON) {
      e.push("manca il giudizio di leggibilità");
    }
    if (b.sufficienza === NON) {
      e.push("manca il giudizio sulla sufficienza delle informazioni");
    }
    return e;
  }

  function salvaPrima(stato, s) {
    var v = voce(stato, s, true);
    if (v.prima_fissata) {
      return [erroreFissato(v)];
    }
    var errori = erroriPrima(v.bozza);
    if (errori.length) {
      return errori;
    }
    var t = ora();
    v.prima = {bozza: copia(v.bozza), salvata_il: t, salvataggi: v.prima ? v.prima.salvataggi + 1 : 1};
    evento(v, "primo_giudizio_salvato");
    if (v.esposta_prima) {
      v.prima_fissata = {quando: t, motivo: "gia_esposta", tramite: v.esposta_prima.tramite};
      evento(v, "primo_giudizio_fissato", "non indipendente: esposta il " + v.esposta_prima.quando);
    }
    return [];
  }

  function bozzaUgualeAPrima(v) {
    return !!(v && v.prima) && canonico(v.bozza) === canonico(v.prima.bozza);
  }

  function bloccanti(ctx, stato, s) {
    var out = {mancanti: [], non_salvate: []};
    ctx.collegati[s.id].forEach(function (c) {
      var w = stato.schede[c.id];
      if (w && (w.prima_fissata || w.esposta_prima)) {
        return;
      }
      if (!w || !w.prima) {
        out.mancanti.push(c.id);
      } else if (!bozzaUgualeAPrima(w)) {
        out.non_salvate.push(c.id);
      }
    });
    return out;
  }

  function puoMostrare(ctx, stato, s) {
    var v = voce(stato, s, false);
    var out = {ok: false, motivo: "", mancanti: [], non_salvate: []};
    if (v && v.soluzione_mostrata_il) {
      out.ok = true;
      out.motivo = "già mostrata";
      return out;
    }
    if (!v || !v.prima) {
      out.motivo = "prima salva il primo giudizio";
      return out;
    }
    if (!v.prima_fissata && !bozzaUgualeAPrima(v)) {
      out.motivo = "hai modifiche non salvate al primo giudizio: salvale prima";
      return out;
    }
    var b = bloccanti(ctx, stato, s);
    out.mancanti = b.mancanti;
    out.non_salvate = b.non_salvate;
    if (b.mancanti.length || b.non_salvate.length) {
      out.motivo = "la soluzione rivela anche le schede collegate: prima salva il loro primo giudizio (" +
        b.mancanti.concat(b.non_salvate).join(", ") + ")";
      return out;
    }
    out.ok = true;
    return out;
  }

  /* Mostra la soluzione: fissa il primo giudizio della scheda e delle schede collegate, registra l'esposizione del
     record e del gruppo. */
  function mostraSoluzione(ctx, stato, s) {
    var p = puoMostrare(ctx, stato, s);
    var v = voce(stato, s, true);
    if (!p.ok) {
      return [p.motivo];
    }
    if (v.soluzione_mostrata_il) {
      return [];
    }
    var t = ora();
    v.soluzione_mostrata_il = t;
    if (!v.prima_fissata) {
      v.prima_fissata = {quando: t, motivo: "soluzione_mostrata", tramite: s.id};
    }
    evento(v, "soluzione_mostrata");
    ["record:" + s.record, "gruppo:" + s.gruppo].forEach(function (k) {
      aggiungiEsposizione(stato, {chiave: k, quando: t, tramite: s.id});
    });
    ctx.collegati[s.id].forEach(function (c) {
      var w = voce(stato, ctx.idx[c.id], true);
      if (w.prima && !w.prima_fissata) {
        w.prima_fissata = {quando: t, motivo: "record_esposto", tramite: s.id};
        evento(w, "primo_giudizio_fissato", "soluzione di " + s.id + " (" + c.collegamento + ")");
      }
    });
    applicaEsposizioni(ctx, stato);
    return [];
  }

  /* ------------------------------------------------------------ secondo passaggio */

  function aggiornaDopo(ctx, stato, s, campo, valore) {
    var v = voce(stato, s, true);
    if (!v.soluzione_mostrata_il) {
      return ["il giudizio finale si compila dopo aver mostrato la soluzione"];
    }
    if (!(campo in v.dopo)) {
      return ["campo sconosciuto: " + campo];
    }
    var m = modulo(ctx, s.id);
    if (SCELTE_DOPO.indexOf(campo) >= 0) {
      if (m.scelte.indexOf(campo) < 0) {
        return ["campo non previsto per questa scheda: " + campo];
      }
      if (ammessi(campo).indexOf(valore) < 0) {
        return ["valore non ammesso: " + valore];
      }
    } else {
      if ((campo === "equivalenza" && !m.lettura) || (campo === "risposte_vere" && !m.diagnostica)) {
        return ["campo non previsto per questa scheda: " + campo];
      }
      valore = String(valore);
    }
    v.dopo[campo] = valore;
    v.aggiornato_il = ora();
    return [];
  }

  function mancantiDopo(ctx, s, v) {
    var m = [];
    if (!v || !v.soluzione_mostrata_il) {
      return ["soluzione non ancora mostrata"];
    }
    modulo(ctx, s.id).scelte.forEach(function (c) {
      if (v.dopo[c] === NON) {
        m.push(c);
      }
    });
    if (SERVE_DIFETTO.indexOf(v.dopo.giudizio_finale) >= 0 && !v.dopo.tipo_difetto.trim()) {
      m.push("tipo_difetto");
    }
    return m;
  }

  function indipendente(ctx, s, v) {
    return !s.controllo_noto && !!(v && v.prima) && !v.esposta_prima;
  }

  function statoScheda(ctx, s, v) {
    var out = {fase: "non_iniziato", incerto: false, non_giudicabile: false, non_indipendente: false, fissata: false};
    if (!v) {
      return out;
    }
    var valori = [v.bozza.leggibilita, v.bozza.sufficienza].concat(SCELTE_DOPO.map(function (c) { return v.dopo[c]; }));
    out.incerto = valori.indexOf("incerto") >= 0;
    out.non_giudicabile = valori.indexOf("non_giudicabile") >= 0;
    out.non_indipendente = !!v.prima && !indipendente(ctx, s, v);
    out.fissata = !!v.prima_fissata;
    if (v.soluzione_mostrata_il) {
      out.fase = mancantiDopo(ctx, s, v).length ? "soluzione_vista" : "completo";
    } else if (v.prima) {
      out.fase = bozzaUgualeAPrima(v) ? "primo_giudizio" : "bozza";
    } else if (canonico(v.bozza) !== canonico(bozzaVuota())) {
      out.fase = "bozza";
    }
    return out;
  }

  function aggiungiTempo(stato, s, ms) {
    var v = voce(stato, s, true);
    v.tempo_ms += Math.max(0, Math.round(ms));
  }

  function riepilogo(ctx, stato) {
    var r = {totale: ctx.ordine.length, completo: 0, soluzione_vista: 0, primo_giudizio: 0, bozza: 0, non_iniziato: 0,
             incerto: 0, non_giudicabile: 0, non_indipendente: 0, esposizioni: stato.esposizioni.length,
             archivio: stato.archivio.length};
    ctx.ordine.forEach(function (id) {
      var x = statoScheda(ctx, ctx.idx[id], stato.schede[id]);
      r[x.fase] += 1;
      r.incerto += x.incerto ? 1 : 0;
      r.non_giudicabile += x.non_giudicabile ? 1 : 0;
      r.non_indipendente += x.non_indipendente ? 1 : 0;
    });
    return r;
  }

  /* ------------------------------------------------------------ esportazione */

  function esportaJSON(ctx, stato) {
    return JSON.stringify(stato, null, 1);
  }

  function cella(valore) {
    return String(valore === null || valore === undefined ? "" : valore).replace(/\\/g, "\\\\").replace(/\t/g, "\\t")
      .replace(/\r?\n/g, "\\n");
  }

  function tsvEsiti(ctx, stato) {
    var righe = [TSV_ESITI.join("\t")];
    ctx.ordine.forEach(function (id) {
      var s = ctx.idx[id], v = stato.schede[id], m = modulo(ctx, id);
      var p = v && v.prima ? v.prima.bozza : null;
      function scelta(c) {
        if (m.scelte.indexOf(c) < 0) {
          return NA;
        }
        return v && v.soluzione_mostrata_il ? v.dopo[c] : NON;
      }
      var riga = {
        scheda: id, record: s.record, controllo_noto: s.controllo_noto ? "si" : "no",
        indipendente: !p ? NON : (indipendente(ctx, s, v) ? "si" : "no"),
        risposta_revisore: p ? (p.non_determinabile ? "non_determinabile" : p.risposta) : NON,
        non_determinabile: p ? (p.non_determinabile ? "si" : "no") : NON,
        leggibilita: p ? p.leggibilita : NON, sufficienza: p ? p.sufficienza : NON, note_prima: p ? p.note : "",
        soluzione_mostrata_il: v && v.soluzione_mostrata_il ? v.soluzione_mostrata_il : "",
        corretta_pixel: scelta("corretta_pixel"), rispondibile: scelta("rispondibile"),
        scrittura_visibile: scelta("scrittura_visibile"), prudenza_eccessiva: scelta("prudenza_eccessiva"),
        giudizio_finale: scelta("giudizio_finale"),
        tipo_difetto: v && v.soluzione_mostrata_il ? v.dopo.tipo_difetto : "",
        equivalenza: v && v.soluzione_mostrata_il && m.lettura ? v.dopo.equivalenza : "",
        risposte_vere: v && v.soluzione_mostrata_il && m.diagnostica ? v.dopo.risposte_vere : "",
        risposta_dopo_soluzione: v && v.soluzione_mostrata_il ? v.dopo.risposta_dopo_soluzione : "",
        note_dopo: v && v.soluzione_mostrata_il ? v.dopo.note : "",
        tempo_s: v ? String(Math.round(v.tempo_ms / 1000)) : "0"
      };
      righe.push(TSV_ESITI.map(function (c) { return cella(riga[c]); }).join("\t"));
    });
    return righe.join("\n") + "\n";
  }

  /* ------------------------------------------------------------ importazione e unione */

  function tipoOk(x, tipo) {
    return tipo === "array" ? Array.isArray(x) : (x !== null && typeof x === tipo && !Array.isArray(x));
  }

  function isoOk(x) {
    return typeof x === "string" && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$/.test(x);
  }

  function chiaviUguali(obj, attese) {
    return tipoOk(obj, "object") && canonico(Object.keys(obj).sort()) === canonico(attese.slice().sort());
  }

  function controllaVoce(ctx, id, v, errori) {
    var nome = "scheda " + id + ": ";
    if (!chiaviUguali(v, ["bozza", "prima", "prima_fissata", "esposta_prima", "soluzione_mostrata_il", "dopo",
                          "tempo_ms", "eventi", "aggiornato_il"])) {
      errori.push(nome + "campi diversi da quelli attesi");
      return;
    }
    function bozzaOk(b) {
      return chiaviUguali(b, Object.keys(bozzaVuota())) && typeof b.risposta === "string" &&
        typeof b.non_determinabile === "boolean" && ammessi("leggibilita").indexOf(b.leggibilita) >= 0 &&
        ammessi("sufficienza").indexOf(b.sufficienza) >= 0 && typeof b.note === "string";
    }
    if (!bozzaOk(v.bozza)) {
      errori.push(nome + "bozza non valida");
    }
    if (v.prima !== null && !(chiaviUguali(v.prima, ["bozza", "salvata_il", "salvataggi"]) && bozzaOk(v.prima.bozza) &&
                              isoOk(v.prima.salvata_il) && typeof v.prima.salvataggi === "number")) {
      errori.push(nome + "primo giudizio non valido");
    }
    if (v.prima_fissata !== null && !(tipoOk(v.prima_fissata, "object") && isoOk(v.prima_fissata.quando) && v.prima)) {
      errori.push(nome + "primo giudizio fissato non valido o senza primo giudizio");
    }
    if (v.esposta_prima !== null && !(tipoOk(v.esposta_prima, "object") && isoOk(v.esposta_prima.quando))) {
      errori.push(nome + "esposizione precedente non valida");
    }
    if (v.soluzione_mostrata_il !== null && !(isoOk(v.soluzione_mostrata_il) && v.prima && v.prima_fissata)) {
      errori.push(nome + "soluzione mostrata senza primo giudizio fissato");
    }
    var d = v.dopo;
    if (!chiaviUguali(d, Object.keys(dopoVuoto()))) {
      errori.push(nome + "giudizio finale con campi diversi da quelli attesi");
    } else {
      SCELTE_DOPO.forEach(function (c) {
        if (ammessi(c).indexOf(d[c]) < 0) {
          errori.push(nome + "valore non ammesso per " + c + ": " + d[c]);
        } else if (d[c] !== NON && (modulo(ctx, id).scelte.indexOf(c) < 0 || !v.soluzione_mostrata_il)) {
          errori.push(nome + c + " compilato dove non è previsto");
        }
      });
      var m = modulo(ctx, id);
      TESTI_DOPO.forEach(function (c) {
        if (typeof d[c] !== "string") {
          errori.push(nome + c + " non è un testo");
        } else if (d[c] && ((c === "equivalenza" && !m.lettura) || (c === "risposte_vere" && !m.diagnostica))) {
          errori.push(nome + c + " compilato dove non è previsto");
        }
      });
    }
    if (typeof v.tempo_ms !== "number" || !Array.isArray(v.eventi) || !isoOk(v.aggiornato_il)) {
      errori.push(nome + "tempo, eventi o data non validi");
    }
  }

  function controllaImport(ctx, testo) {
    var dati, errori = [];
    try {
      dati = JSON.parse(testo);
    } catch (e) {
      return {errori: ["il file non è un JSON valido"], stato: null};
    }
    if (!tipoOk(dati, "object") || dati.schema !== SCHEMA_STATO) {
      return {errori: ["formato diverso da " + SCHEMA_STATO], stato: null};
    }
    var p = dati.pacchetto || {};
    if (p.id !== ctx.pkg.id || p.versione !== ctx.pkg.versione || p.impronta !== ctx.pkg.impronta) {
      errori.push("il file è di un altro pacchetto o di un'altra versione (" + [p.id, p.versione,
        String(p.impronta).slice(0, 12)].join(", ") + ")");
    }
    if (!tipoOk(dati.schede, "object") || !Array.isArray(dati.esposizioni) || !Array.isArray(dati.archivio)) {
      errori.push("schede, esposizioni o archivio mancanti");
      return {errori: errori, stato: null};
    }
    Object.keys(dati.schede).forEach(function (id) {
      if (!ctx.idx[id]) {
        errori.push("scheda sconosciuta: " + id);
      } else {
        controllaVoce(ctx, id, dati.schede[id], errori);
      }
    });
    dati.esposizioni.forEach(function (e, i) {
      var noto = tipoOk(e, "object") && ctx.colpiteDi[e.chiave] && ctx.idx[e.tramite] && isoOk(e.quando);
      if (!noto) {
        errori.push("esposizione " + (i + 1) + " non valida");
      }
    });
    if (!(dati.posizione === null || ctx.idx[dati.posizione])) {
      errori.push("posizione sconosciuta: " + dati.posizione);
    }
    return {errori: errori, stato: errori.length ? null : dati};
  }

  function piuAntica(a, b) {
    return !a ? b : !b ? a : (a < b ? a : b);
  }

  /* Unisce allo stato corrente uno stato importato (o salvato da un'altra scheda del browser). La conoscenza non si
     perde mai: esposizioni unite, un primo giudizio fissato vince su uno non fissato, fra due fissati diversi vince il
     più antico, la soluzione mostrata resta alla data più antica. Il lavoro in conflitto (stessa scheda diversa, senza
     fissazioni che decidano) segue il modo: "browser" o "senza_scelta" tengono lo stato corrente, "file" e
     "sostituisci" quello importato; "sostituisci" toglie anche le schede presenti solo nel browser senza un primo
     giudizio fissato. Ogni versione scartata va nell'archivio. */
  function unisci(ctx, corrente, importato, modo) {
    var out = copia(corrente);
    var r = {nuove: 0, esposizioni_nuove: 0, archiviate: 0, conflitti: [], decise_da_fissazione: []};
    importato.esposizioni.forEach(function (e) {
      if (aggiungiEsposizione(out, e)) {
        r.esposizioni_nuove += 1;
      }
    });
    importato.archivio.forEach(function (a) {
      var k = a.id + "|" + canonico(a.contenuto);
      if (!out.archivio.some(function (x) { return x.id + "|" + canonico(x.contenuto) === k; })) {
        out.archivio.push(copia(a));
      }
    });
    var preferisciFile = modo === "file" || modo === "sostituisci";
    Object.keys(importato.schede).forEach(function (id) {
      var a = out.schede[id], b = importato.schede[id];
      if (!a) {
        out.schede[id] = copia(b);
        r.nuove += 1;
        return;
      }
      if (canonico(a) === canonico(b)) {
        return;
      }
      var fa = a.prima_fissata, fb = b.prima_fissata, vince;
      if (fa && !fb) {
        vince = "corrente";
      } else if (fb && !fa) {
        vince = "importato";
      } else if (fa && fb && canonico(a.prima) !== canonico(b.prima)) {
        vince = fa.quando <= fb.quando ? "corrente" : "importato";
      } else {
        vince = preferisciFile ? "importato" : "corrente";
        r.conflitti.push(id);
      }
      if ((fa || fb) && r.conflitti.indexOf(id) < 0) {
        r.decise_da_fissazione.push(id);
      }
      var tenuta = vince === "importato" ? copia(b) : a, scartata = vince === "importato" ? a : b;
      tenuta.soluzione_mostrata_il = piuAntica(a.soluzione_mostrata_il, b.soluzione_mostrata_il);
      if (tenuta.soluzione_mostrata_il && !tenuta.prima_fissata) {
        tenuta.prima_fissata = fa || fb;
      }
      if (archivia(out, id, vince === "importato" ? "corrente" : "importato",
                   r.conflitti.indexOf(id) >= 0 ? "conflitto: " + (modo || "senza_scelta") : "primo giudizio fissato",
                   scartata)) {
        r.archiviate += 1;
      }
      out.schede[id] = tenuta;
    });
    if (modo === "sostituisci") {
      Object.keys(out.schede).forEach(function (id) {
        if (!importato.schede[id] && !out.schede[id].prima_fissata) {
          if (archivia(out, id, "corrente", "sostituito dal file", out.schede[id])) {
            r.archiviate += 1;
          }
          delete out.schede[id];
        }
      });
    }
    out.aggiornato_il = ora();
    applicaEsposizioni(ctx, out);
    r.stato = out;
    return r;
  }

  function impronteConoscenza(stato) {
    var f = {};
    Object.keys(stato.schede).sort().forEach(function (id) {
      var v = stato.schede[id];
      f[id] = [v.prima_fissata, v.soluzione_mostrata_il, v.esposta_prima];
    });
    return canonico({esposizioni: stato.esposizioni.map(chiaveEsposizione).sort(), fissazioni: f});
  }

  root.Logica = {
    SCHEMA_STATO: SCHEMA_STATO, NON: NON, NA: NA, VALORI: VALORI, AIUTI: AIUTI, TIPI_DIFETTO: TIPI_DIFETTO,
    SCELTE_DOPO: SCELTE_DOPO, TSV_ESITI: TSV_ESITI, ora: ora, copia: copia, canonico: canonico,
    contesto: contesto, modulo: modulo, nuovoStato: nuovoStato, voce: voce, aggiornaBozza: aggiornaBozza,
    erroriPrima: erroriPrima, salvaPrima: salvaPrima, bozzaUgualeAPrima: bozzaUgualeAPrima, bloccanti: bloccanti,
    puoMostrare: puoMostrare,
    mostraSoluzione: mostraSoluzione, aggiornaDopo: aggiornaDopo, mancantiDopo: mancantiDopo,
    indipendente: indipendente, statoScheda: statoScheda, aggiungiTempo: aggiungiTempo, riepilogo: riepilogo,
    esportaJSON: esportaJSON, tsvEsiti: tsvEsiti, controllaImport: controllaImport, unisci: unisci,
    applicaEsposizioni: applicaEsposizioni, impronteConoscenza: impronteConoscenza, primaEsposizione: primaEsposizione
  };
})(window);
