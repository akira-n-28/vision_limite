/* Logica della revisione offline del campione-2 dei sintetici v4: stato, primo e secondo passaggio, coppie,
   collegamenti fra unità ed esposizioni dei record, decontaminazione, esportazione, importazione e unione degli stati.
   Nessun accesso al DOM né alla rete: la usano index.html (app.js) e verifica.html (test.js).

   Esposizioni. Mostrare la soluzione di un'unità espone i suoi record: target e opzioni, tutte le QA con le risposte,
   specifica, layout e riga del checker. Lo stesso materiale rivela le risposte delle altre unità che mostrano lo stesso
   record o un record con la stessa scena (specifica senza chiave, versione del generatore e stile). Per questo:
   - la soluzione si mostra solo quando tutte le unità collegate hanno un primo giudizio salvato;
   - al momento della rivelazione il primo giudizio di queste unità resta fissato (prima_fissata);
   - ogni esposizione si registra in stato.esposizioni (record, quando, tramite quale unità) e non si cancella più;
   - un'unità esposta prima di avere un primo giudizio salvato è segnata (esposta_senza_primo_giudizio): il suo primo
     giudizio, quando lo salverà, non è indipendente;
   - unendo due stati (importazione, sostituzione, ripresa nel browser) esposizioni e primi giudizi fissati restano; le
     versioni scartate vanno in stato.archivio, mai perse in silenzio. */
(function (root) {
  "use strict";

  var SCHEMA_STATO = "synth-v4-revisione-avanzamento/2";
  var SCHEMA_V1 = "synth-v4-revisione-avanzamento/1";
  var NON = "non_compilato";
  var STILE = ["lw", "font", "font_family", "width", "height"];  // synth_candidate.STYLE: stile, non stato della scena

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
    decisiva: "La corrispondenza che mostra lo stesso contenuto.",
    collegate: "Queste unità mostrano lo stesso record o un record con la stessa scena. Il materiale della soluzione " +
               "(QA con le risposte, specifica, layout, checker) rivela anche le loro risposte: prima di mostrarlo " +
               "salva il loro primo giudizio, che da quel momento resta fissato."
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
  var MOTIVI_FISSAZIONE = ["soluzione_mostrata", "record_esposto", "record_gia_esposto"];
  var COLLEGAMENTI = ["stessa_unita", "stesso_record", "stessa_scena"];
  var NOMI_COLLEGAMENTO = {stessa_unita: "questa unità", stesso_record: "stesso record", stessa_scena: "stessa scena"};

  function ammessi(nome) {
    return VALORI[nome].map(function (x) { return x[0]; });
  }

  function ora() {
    return new Date().toISOString();
  }

  function haValore(lista, valore) {
    return lista.indexOf(valore) >= 0;
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

  /* ------------------------------------------------------------ indice e collegamenti */

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

  /* Stato della scena di un record, come synth_candidate.scene_state: la specifica senza chiave, versione del
     generatore e stile, in forma canonica. Null se il materiale del record manca. */
  function chiaveScena(evi, record) {
    var e = evi && evi[record];
    var spec;
    if (!e) {
      return null;
    }
    try {
      spec = JSON.parse(e.json).spec;
    } catch (err) {
      return null;
    }
    if (!spec || typeof spec !== "object") {
      return null;
    }
    var stato = {};
    Object.keys(spec).forEach(function (k) {
      if (k !== "key" && k !== "generator_version" && STILE.indexOf(k) < 0) {
        stato[k] = spec[k];
      }
    });
    return canonico(stato);
  }

  var cacheContesto = {pkg: null, evi: null, ctx: null};

  /* Contesto della revisione: indice del pacchetto e collegamenti fra unità.
     - recordDi[unità]: i record che l'unità mostra (e che la sua soluzione espone);
     - unitaDi[record]: le unità che mostrano il record;
     - colpiteDi[record]: le unità di cui l'esposizione del record rivela la risposta (stesso record o stessa scena);
     - collegati[unità]: le altre unità colpite dalla sua soluzione, con record e tipo di collegamento. */
  function contesto(pkg, evi) {
    if (cacheContesto.ctx && cacheContesto.pkg === pkg && cacheContesto.evi === evi) {
      return cacheContesto.ctx;
    }
    var ctx = {pkg: pkg, idx: indice(pkg), recordDi: {}, unitaDi: {}, scenaDi: {}, recordDiScena: {}, colpiteDi: {},
               collegati: {}, scene_note: !!evi};
    var unita = pkg.campione.concat(pkg.mirati);
    unita.forEach(function (u) {
      var recs = [];
      u.immagini.forEach(function (im) { aggiungiUnico(recs, im.record); });
      ctx.recordDi[u.id] = recs;
      recs.forEach(function (r) { (ctx.unitaDi[r] = ctx.unitaDi[r] || []).push(u.id); });
    });
    Object.keys(ctx.unitaDi).forEach(function (r) {
      var k = chiaveScena(evi, r);
      if (k !== null) {
        ctx.scenaDi[r] = k;
        (ctx.recordDiScena[k] = ctx.recordDiScena[k] || []).push(r);
      }
    });
    Object.keys(ctx.unitaDi).forEach(function (r) {
      var out = [], viste = {};
      var simili = ctx.scenaDi[r] !== undefined ? ctx.recordDiScena[ctx.scenaDi[r]] : [r];
      [r].concat(simili.filter(function (x) { return x !== r; })).forEach(function (x) {
        ctx.unitaDi[x].forEach(function (id) {
          if (!viste[id]) {
            viste[id] = true;
            out.push({id: id, record: x, collegamento: x === r ? "stesso_record" : "stessa_scena"});
          }
        });
      });
      ctx.colpiteDi[r] = out;
    });
    unita.forEach(function (u) {
      var out = [], pos = {};
      ctx.recordDi[u.id].forEach(function (r) {
        ctx.colpiteDi[r].forEach(function (c) {
          if (c.id === u.id) {
            return;
          }
          if (pos[c.id] === undefined) {
            pos[c.id] = out.length;
            out.push({id: c.id, record_esposti: [], record_collegati: [], collegamento: []});
          }
          var o = out[pos[c.id]];
          aggiungiUnico(o.record_esposti, r);
          aggiungiUnico(o.record_collegati, c.record);
          aggiungiUnico(o.collegamento, c.collegamento);
        });
      });
      ctx.collegati[u.id] = out;
    });
    cacheContesto = {pkg: pkg, evi: evi, ctx: ctx};
    return ctx;
  }

  function descriviCollegamento(c) {
    return c.collegamento.map(function (k) { return NOMI_COLLEGAMENTO[k]; }).join(", ") + ": " +
      c.record_collegati.join(", ") + " (record esposti: " + c.record_esposti.join(", ") + ")";
  }

  /* ------------------------------------------------------------ stato */

  function nuovoStato(pkg) {
    var t = ora();
    return {schema: SCHEMA_STATO, pacchetto: {id: pkg.id, versione: pkg.versione, impronta: pkg.impronta},
            revisore: "", creato_il: t, aggiornato_il: t, unita: {}, decontaminazione: {}, posizione: null,
            esposizioni: [], archivio: [], migrazioni: []};
  }

  function evento(voce, azione, dettaglio) {
    var e = {quando: ora(), azione: azione};
    if (dettaglio !== undefined) {
      e.dettaglio = dettaglio;
    }
    voce.eventi.push(e);
    voce.aggiornato_il = e.quando;
  }

  function chiaveArchivio(a) {
    return a.sezione + "|" + a.id + "|" + canonico(a.contenuto);
  }

  /* Conserva nell'archivio dello stato una versione che non resta in vigore. Restituisce true se è nuova. */
  function archivia(stato, sezione, id, origine, motivo, contenuto) {
    var a = {quando: ora(), sezione: sezione, id: id, origine: origine, motivo: motivo, contenuto: copia(contenuto)};
    var k = chiaveArchivio(a);
    if (stato.archivio.some(function (x) { return chiaveArchivio(x) === k; })) {
      return false;
    }
    stato.archivio.push(a);
    return true;
  }

  function chiaveEsposizione(e) {
    return e.record + "|" + e.tramite + "|" + e.quando;
  }

  function aggiungiEsposizione(stato, e) {
    var k = chiaveEsposizione(e);
    if (stato.esposizioni.some(function (x) { return chiaveEsposizione(x) === k; })) {
      return false;
    }
    stato.esposizioni.push({record: e.record, quando: e.quando, tramite: e.tramite, origine: e.origine});
    return true;
  }

  /* ------------------------------------------------------------ unità del campione e casi mirati */

  function bozzaVuota() {
    return {risposta: "", non_riesco: false, leggibilita: NON, note: ""};
  }

  function dopoVuoto() {
    return {corretta: NON, coerente_con_annotazioni: NON, rispondibile_da_immagine: NON, gravita: NON,
            tipo_difetto: "", note: "", risposta_dopo_soluzione: ""};
  }

  function dopoVuotoUnita(unit) {
    var d = {varianti: {}};
    unit.varianti.forEach(function (k) { d.varianti[k] = dopoVuoto(); });
    if (unit.coppia) {
      d.coppia = {contratto_coppia: NON, gravita: NON, tipo_difetto: "", note: ""};
    }
    return d;
  }

  function voce(stato, unit, crea) {
    var v = stato.unita[unit.id];
    if (!v && crea) {
      v = {tipo: "unita", varianti: {}, bozza: {}, prima: null, prima_fissata: null, esposta_senza_primo_giudizio: null,
           soluzione_mostrata_il: null, dopo: dopoVuotoUnita(unit), tempo_ms: 0, eventi: [], aggiornato_il: ora()};
      unit.varianti.forEach(function (k) {
        v.varianti[k] = {record: unit.record_per_variante[k]};
        v.bozza[k] = bozzaVuota();
      });
      stato.unita[unit.id] = v;
    }
    return v || null;
  }

  function descriviFissazione(f) {
    if (f.motivo === "soluzione_mostrata") {
      return "fissato il " + f.quando + " mostrando la soluzione di questa unità";
    }
    var come = NOMI_COLLEGAMENTO[f.collegamento] || f.collegamento;
    if (f.motivo === "record_esposto") {
      return "fissato il " + f.quando + ": la soluzione di " + f.tramite + " ha esposto " + f.record.join(", ") +
        " (" + come + ")";
    }
    return "fissato al salvataggio, il " + f.quando + ": " + f.record.join(", ") + " era già esposto tramite " +
      f.tramite + " (" + come + ")";
  }

  function descriviEsposizione(e) {
    return e.record.join(", ") + " esposto il " + e.quando + " tramite " + e.tramite + " (" +
      (NOMI_COLLEGAMENTO[e.collegamento] || e.collegamento) + ")";
  }

  function erroreFissato(v) {
    return "il primo giudizio è fissato (" + descriviFissazione(v.prima_fissata) + "): non si modifica più; i " +
      "ripensamenti vanno nel giudizio finale";
  }

  function aggiornaBozza(stato, unit, variante, campo, valore) {
    var v = voce(stato, unit, true);
    if (v.prima_fissata) {
      return [erroreFissato(v)];
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

  /* Salva il primo giudizio. Se un record dell'unità è già stato esposto (esposta_senza_primo_giudizio), il primo
     giudizio non è indipendente e resta fissato subito. */
  function salvaPrima(stato, unit) {
    var v = voce(stato, unit, true);
    if (v.prima_fissata) {
      return [erroreFissato(v)];
    }
    var errori = erroriPrima(unit, v.bozza);
    if (errori.length) {
      return errori;
    }
    var t = ora();
    v.prima = {varianti: copia(v.bozza), salvata_il: t, salvataggi: v.prima ? v.prima.salvataggi + 1 : 1};
    evento(v, "primo_giudizio_salvato");
    var e = v.esposta_senza_primo_giudizio;
    if (e) {
      v.prima_fissata = {quando: t, motivo: "record_gia_esposto", tramite: e.tramite, record: e.record.slice(),
                         collegamento: e.collegamento};
      evento(v, "primo_giudizio_fissato", "non indipendente: " + descriviEsposizione(e));
    }
    return [];
  }

  function bozzaUgualeAPrima(v) {
    return !!(v && v.prima) && canonico(v.bozza) === canonico(v.prima.varianti);
  }

  /* Unità collegate che impediscono di mostrare la soluzione: senza primo giudizio salvato o con modifiche non
     salvate. Non bloccano quelle già fissate e quelle già esposte in precedenza (il loro giudizio non è comunque più
     indipendente). */
  function bloccanti(ctx, stato, unit) {
    var out = {mancanti: [], non_salvate: []};
    (ctx.collegati[unit.id] || []).forEach(function (c) {
      var w = stato.unita[c.id];
      if (w && (w.prima_fissata || w.esposta_senza_primo_giudizio)) {
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

  function puoMostrare(ctx, stato, unit) {
    var v = voce(stato, unit, false);
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
    var b = bloccanti(ctx, stato, unit);
    out.mancanti = b.mancanti;
    out.non_salvate = b.non_salvate;
    if (b.mancanti.length || b.non_salvate.length) {
      out.motivo = "la soluzione rivela anche le unità collegate: prima salva il loro primo giudizio" +
        (b.mancanti.length ? "; senza primo giudizio: " + b.mancanti.join(", ") : "") +
        (b.non_salvate.length ? "; con modifiche non salvate: " + b.non_salvate.join(", ") : "");
      return out;
    }
    out.ok = true;
    return out;
  }

  /* Prima esposizione registrata che riguarda ciascuna unità: {quando, tramite, record, collegamento}. */
  function primeEsposizioni(ctx, stato) {
    var prime = {};
    stato.esposizioni.slice().sort(function (a, b) {
      return a.quando < b.quando ? -1 : a.quando > b.quando ? 1 : chiaveEsposizione(a) < chiaveEsposizione(b) ? -1 : 1;
    }).forEach(function (e) {
      (ctx.colpiteDi[e.record] || []).forEach(function (c) {
        if (!prime[c.id]) {
          prime[c.id] = {quando: e.quando, tramite: e.tramite, record: [e.record],
                         collegamento: c.id === e.tramite ? "stessa_unita" : c.collegamento};
        }
      });
    });
    return prime;
  }

  /* Applica le esposizioni registrate allo stato e ne ricava quelle implicite nelle soluzioni mostrate (stati del
     formato 1). Blocco e indipendenza sono due cose distinte:
     - indipendenza: per ogni unità conta la prima esposizione che la riguarda (registro, o l'esposizione già annotata
       nell'unità se è più antica), anche quando il primo giudizio è già fissato. Primo giudizio salvato non dopo
       quell'esposizione: indipendente. Nessun primo giudizio, o salvato dopo: esposta_senza_primo_giudizio, con la
       provenienza della prima esposizione;
     - blocco: un primo giudizio salvato e colpito da un'esposizione resta fissato. Il contenuto non cambia mai; la
       provenienza del blocco (prima_fissata) passa all'evento più antico che lo avrebbe fissato: l'esposizione, se il
       giudizio è indipendente, altrimenti il salvataggio stesso.
     Ogni cambio di classificazione o di provenienza è un evento dell'unità. Una bozza diversa dal primo giudizio fissato
     va nell'archivio e la bozza torna uguale al primo giudizio. Applicata due volte, la seconda non cambia nulla. */
  function applicaEsposizioni(ctx, stato) {
    var out = {fissate: [], esposte: [], riclassificate: [], indipendenti: [], provenienze: [], archiviate: 0,
               derivate: 0};
    Object.keys(stato.unita).forEach(function (id) {
      var v = stato.unita[id];
      if (!v.soluzione_mostrata_il || !ctx.recordDi[id]) {
        return;
      }
      ctx.recordDi[id].forEach(function (r) {
        if (aggiungiEsposizione(stato, {record: r, quando: v.soluzione_mostrata_il, tramite: id,
                                        origine: "soluzione_mostrata"})) {
          out.derivate += 1;
        }
      });
      if (!v.prima_fissata) {
        v.prima_fissata = {quando: v.soluzione_mostrata_il, motivo: "soluzione_mostrata", tramite: id,
                           record: ctx.recordDi[id].slice(), collegamento: "stessa_unita"};
      }
    });
    var prime = primeEsposizioni(ctx, stato);
    var ids = Object.keys(prime);
    Object.keys(stato.unita).forEach(function (id) {
      if (stato.unita[id].esposta_senza_primo_giudizio && !prime[id]) {
        ids.push(id);
      }
    });
    ids.forEach(function (id) {
      var v = voce(stato, ctx.idx.unita[id], true);
      var annotata = v.esposta_senza_primo_giudizio;
      var e = prime[id] && !(annotata && annotata.quando < prime[id].quando) ? prime[id] : copia(annotata);
      var indipendente = !!v.prima && v.prima.salvata_il <= e.quando;
      if (indipendente) {
        if (annotata) {  // annotazione smentita dalle date: il giudizio è stato salvato prima di ogni esposizione
          v.esposta_senza_primo_giudizio = null;
          evento(v, "riclassificata_indipendente", "primo giudizio salvato il " + v.prima.salvata_il +
                 ", non dopo la prima esposizione (" + descriviEsposizione(e) + ")");
          out.indipendenti.push(id);
        }
      } else if (!annotata || canonico(annotata) !== canonico(e)) {
        v.esposta_senza_primo_giudizio = copia(e);
        evento(v, annotata ? "esposizione_precedente_aggiornata" : "esposta_senza_primo_giudizio",
               descriviEsposizione(e) + (v.prima ? "; primo giudizio salvato dopo, il " + v.prima.salvata_il : ""));
        (annotata || v.prima_fissata ? out.riclassificate : out.esposte).push(id);
      }
      if (!v.prima) {
        return;
      }
      var propria = e.tramite === id;
      var blocco = indipendente ?
        {quando: e.quando, motivo: propria ? "soluzione_mostrata" : "record_esposto", tramite: e.tramite,
         record: propria ? ctx.recordDi[id].slice() : e.record.slice(), collegamento: e.collegamento} :
        {quando: v.prima.salvata_il, motivo: "record_gia_esposto", tramite: e.tramite, record: e.record.slice(),
         collegamento: e.collegamento};
      if (!v.prima_fissata) {
        v.prima_fissata = blocco;
        evento(v, "primo_giudizio_fissato", descriviFissazione(blocco));
        out.fissate.push(id);
      } else if (blocco.quando < v.prima_fissata.quando) {  // stesso contenuto, bloccato da prima
        v.prima_fissata = blocco;
        evento(v, "provenienza_del_blocco_aggiornata", descriviFissazione(blocco));
        out.provenienze.push(id);
      }
    });
    Object.keys(stato.unita).forEach(function (id) {
      var v = stato.unita[id];
      if (v.prima_fissata && v.prima && !bozzaUgualeAPrima(v)) {
        if (archivia(stato, "unita", id, "bozza", "bozza diversa dal primo giudizio fissato (modifiche non salvate " +
                     "al momento della fissazione): non indipendente se successiva all'esposizione", v.bozza)) {
          out.archiviate += 1;
        }
        v.bozza = copia(v.prima.varianti);
      }
    });
    return out;
  }

  function mostraSoluzione(ctx, stato, unit) {
    var p = puoMostrare(ctx, stato, unit);
    if (!p.ok) {
      return [p.motivo];
    }
    var v = voce(stato, unit, true);
    if (v.soluzione_mostrata_il) {
      return [];
    }
    var t = ora();
    v.soluzione_mostrata_il = t;
    if (!v.prima_fissata) {
      v.prima_fissata = {quando: t, motivo: "soluzione_mostrata", tramite: unit.id, record: ctx.recordDi[unit.id].slice(),
                         collegamento: "stessa_unita"};
    }
    evento(v, "soluzione_mostrata", "record esposti: " + ctx.recordDi[unit.id].join(", "));
    ctx.recordDi[unit.id].forEach(function (r) {
      aggiungiEsposizione(stato, {record: r, quando: t, tramite: unit.id, origine: "soluzione_mostrata"});
    });
    applicaEsposizioni(ctx, stato);
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
      if (campo === "note" || campo === "tipo_difetto" || campo === "risposta_dopo_soluzione") {
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

  /* fase: non_iniziato, bozza, primo_giudizio (salvato, soluzione nascosta), soluzione_vista, completo.
     fissata: il primo giudizio non si modifica più; non_indipendente: un suo record è stato esposto prima che il
     primo giudizio fosse salvato. */
  function statoUnita(unit, v) {
    var valori = valoriDi(v);
    var out = {fase: "non_iniziato", mancanti: [], incerto: haValore(valori, "incerto"),
               non_giudicabile: haValore(valori, "non_giudicabile"), non_verificabile: haValore(valori, "non_verificabile"),
               fissata: !!(v && v.prima_fissata), non_indipendente: !!(v && v.esposta_senza_primo_giudizio)};
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

  function riepilogo(ctx, stato) {
    var pkg = ctx.pkg;
    var out = {};
    ["campione", "mirati"].forEach(function (sezione) {
      var tot = {totale: 0, gruppi: {}};
      pkg[sezione].forEach(function (u) {
        var s = statoUnita(u, stato.unita[u.id]);
        var g = tot.gruppi[u.gruppo] = tot.gruppi[u.gruppo] || {totale: 0};
        [tot, g].forEach(function (c) {
          c.totale += 1;
          c[s.fase] = (c[s.fase] || 0) + 1;
          ["incerto", "non_giudicabile", "non_verificabile", "non_indipendente"].forEach(function (f) {
            if (s[f]) {
              c[f] = (c[f] || 0) + 1;
            }
          });
          if (s.fissata && s.fase === "primo_giudizio") {
            c.fissate_senza_soluzione = (c.fissate_senza_soluzione || 0) + 1;
          }
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
    var record = {};
    stato.esposizioni.forEach(function (e) { record[e.record] = true; });
    out.esposizioni = {eventi: stato.esposizioni.length, record: Object.keys(record).length,
                       archivio: stato.archivio.length};
    return out;
  }

  /* ------------------------------------------------------------ esportazione */

  function oggettoEsportato(ctx, stato) {
    var pkg = ctx.pkg;
    return {schema: SCHEMA_STATO, pacchetto: {id: pkg.id, versione: pkg.versione, impronta: pkg.impronta},
            revisore: stato.revisore, creato_il: stato.creato_il, aggiornato_il: stato.aggiornato_il,
            esportato_il: ora(), posizione: stato.posizione, unita: stato.unita,
            decontaminazione: stato.decontaminazione, esposizioni: stato.esposizioni, archivio: stato.archivio,
            migrazioni: stato.migrazioni, riepilogo: riepilogo(ctx, stato)};
  }

  function esportaJSON(ctx, stato) {
    return JSON.stringify(oggettoEsportato(ctx, stato), null, 1) + "\n";
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

  /* esiti.tsv con le colonne del modulo. Una risposta non indipendente (record esposto prima del primo giudizio) non
     va nella colonna della risposta del revisore: lì c'è "non_indipendente", e la risposta con il motivo va nelle
     note. */
  function tsvEsiti(ctx, stato) {
    var idx = ctx.idx;
    var righe = [TSV_ESITI.join("\t")];
    ctx.pkg.righe_esiti.forEach(function (r) {
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
        var note = d ? d.note : "";
        if (p && v.esposta_senza_primo_giudizio) {
          note = "[primo giudizio non indipendente: " + descriviEsposizione(v.esposta_senza_primo_giudizio) +
            "] risposta: " + risposta + (note ? "\n" + note : "");
          risposta = "non_indipendente";
        }
        row = [r.unita, r.variante, r.record, risposta, d ? d.corretta : NON, d ? d.coerente_con_annotazioni : NON,
               d ? d.rispondibile_da_immagine : NON, "non_applicabile", "non_applicabile", d ? d.gravita : NON,
               d ? d.tipo_difetto : "", tempo, note];
      }
      righe.push(rigaTsv(row));
    });
    return righe.join("\n") + "\n";
  }

  function tsvDecontaminazione(ctx, stato) {
    var idx = ctx.idx;
    var righe = [TSV_DECONTAMINAZIONE.join("\t")];
    ctx.pkg.righe_esiti_decontaminazione.forEach(function (r) {
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

  function rigaPrimaValida(p) {
    return chiaviUguali(p, ["risposta", "non_riesco", "leggibilita", "note"]) && typeof p.risposta === "string"
      && typeof p.note === "string" && typeof p.non_riesco === "boolean" && haValore(ammessi("leggibilita"), p.leggibilita);
  }

  function recordValidi(ctx, lista) {
    return Array.isArray(lista) && lista.length > 0 && lista.every(function (r) {
      return typeof r === "string" && !!ctx.unitaDi[r]; });
  }

  function fissazioneValida(ctx, f) {
    return chiaviUguali(f, ["quando", "motivo", "tramite", "record", "collegamento"]) && typeof f.quando === "string"
      && haValore(MOTIVI_FISSAZIONE, f.motivo) && !!ctx.idx.unita[f.tramite] && recordValidi(ctx, f.record)
      && haValore(COLLEGAMENTI, f.collegamento);
  }

  function esposizioneUnitaValida(ctx, e) {
    return chiaviUguali(e, ["quando", "tramite", "record", "collegamento"]) && typeof e.quando === "string"
      && !!ctx.idx.unita[e.tramite] && recordValidi(ctx, e.record) && haValore(COLLEGAMENTI, e.collegamento);
  }

  function controllaVoceUnita(ctx, unit, v, errori, formato) {
    var dove = unit.id + ": ";
    var tutte = ["tipo", "varianti", "bozza", "prima", "soluzione_mostrata_il", "dopo", "tempo_ms", "eventi",
                 "aggiornato_il"];
    if (formato === 2) {
      tutte = tutte.concat(["prima_fissata", "esposta_senza_primo_giudizio"]);
    }
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
      if (!rigaPrimaValida(v.bozza[k])) {
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
          || typeof v.prima.salvata_il !== "string" || typeof v.prima.salvataggi !== "number"
          || unit.varianti.some(function (k) { return !rigaPrimaValida(v.prima.varianti[k]); })) {
        errori.push(dove + "primo giudizio non valido");
      } else if (erroriPrima(unit, v.prima.varianti).length) {
        errori.push(dove + "primo giudizio incompleto");
      }
    }
    if (v.soluzione_mostrata_il !== null && (typeof v.soluzione_mostrata_il !== "string" || v.prima === null)) {
      errori.push(dove + "soluzione mostrata senza un primo giudizio salvato");
    }
    if (formato === 2) {
      if (v.prima_fissata !== null && (!fissazioneValida(ctx, v.prima_fissata) || v.prima === null)) {
        errori.push(dove + "primo giudizio fissato non valido o senza primo giudizio");
      }
      if (v.soluzione_mostrata_il !== null && v.prima_fissata === null) {
        errori.push(dove + "soluzione mostrata con il primo giudizio non fissato");
      }
      if (v.esposta_senza_primo_giudizio !== null && !esposizioneUnitaValida(ctx, v.esposta_senza_primo_giudizio)) {
        errori.push(dove + "esposizione dell'unità non valida");
      }
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

  function controllaRegistri(ctx, dati, errori) {
    if (!tipoOk(dati.esposizioni, "array") || !tipoOk(dati.archivio, "array") || !tipoOk(dati.migrazioni, "array")) {
      errori.push("mancano i registri esposizioni, archivio o migrazioni");
      return;
    }
    dati.esposizioni.forEach(function (e, i) {
      if (!chiaviUguali(e, ["record", "quando", "tramite", "origine"]) || typeof e.quando !== "string"
          || typeof e.origine !== "string" || !ctx.unitaDi[e.record] || !ctx.recordDi[e.tramite]
          || ctx.recordDi[e.tramite].indexOf(e.record) < 0) {
        errori.push("esposizione " + i + " non valida o con record e unità che non si corrispondono");
      }
    });
    dati.archivio.forEach(function (a, i) {
      if (!chiaviUguali(a, ["quando", "sezione", "id", "origine", "motivo", "contenuto"]) || typeof a.quando !== "string"
          || (a.sezione !== "unita" && a.sezione !== "decontaminazione") || typeof a.id !== "string"
          || typeof a.origine !== "string" || typeof a.motivo !== "string") {
        errori.push("voce " + i + " dell'archivio non valida");
      }
    });
    dati.migrazioni.forEach(function (m, i) {
      if (!tipoOk(m, "object") || typeof m.da !== "string" || typeof m.a !== "string" || typeof m.quando !== "string") {
        errori.push("migrazione " + i + " non valida");
      }
    });
  }

  /* Controlla un file di avanzamento e restituisce {errori, stato, migrazione}. Il formato 1 (pacchetto pubblicato
     con a9252b5) si accetta con una migrazione esplicita: le soluzioni già mostrate diventano esposizioni dei loro
     record, e le unità collegate sono fissate (primo giudizio salvato prima) o segnate come non indipendenti. */
  function controllaImport(ctx, testo) {
    var pkg = ctx.pkg;
    var errori = [];
    var dati;
    try {
      dati = JSON.parse(testo);
    } catch (e) {
      return {errori: ["il file non è un JSON valido: " + e.message], stato: null, migrazione: null};
    }
    if (!tipoOk(dati, "object") || (dati.schema !== SCHEMA_STATO && dati.schema !== SCHEMA_V1)) {
      return {errori: ["formato del file " + (dati && dati.schema) + " sconosciuto: si leggono " + SCHEMA_STATO +
                       " e, con una migrazione, " + SCHEMA_V1], stato: null, migrazione: null};
    }
    var formato = dati.schema === SCHEMA_STATO ? 2 : 1;
    var p = dati.pacchetto || {};
    if (p.id !== pkg.id || p.versione !== pkg.versione || p.impronta !== pkg.impronta) {
      return {errori: ["il file appartiene a un altro pacchetto o a un'altra versione: " + p.id + " v" + p.versione +
                       ", impronta " + String(p.impronta).slice(0, 12) + " (questo pacchetto: " + pkg.id + " v" +
                       pkg.versione + ", " + pkg.impronta.slice(0, 12) + ")"], stato: null, migrazione: null};
    }
    if (!tipoOk(dati.unita, "object") || !tipoOk(dati.decontaminazione, "object")) {
      return {errori: ["mancano le sezioni unita o decontaminazione"], stato: null, migrazione: null};
    }
    var idx = ctx.idx;
    Object.keys(dati.unita).forEach(function (id) {
      if (!idx.unita[id]) {
        errori.push(id + ": unità sconosciuta in questo pacchetto");
      } else {
        controllaVoceUnita(ctx, idx.unita[id], dati.unita[id], errori, formato);
      }
    });
    Object.keys(dati.decontaminazione).forEach(function (id) {
      if (!idx.decontaminazione[id]) {
        errori.push(id + ": candidato sconosciuto in questo pacchetto");
      } else {
        controllaVoceDecontam(idx.decontaminazione[id].d, dati.decontaminazione[id], errori);
      }
    });
    if (formato === 2) {
      controllaRegistri(ctx, dati, errori);
    }
    if (errori.length) {
      return {errori: errori, stato: null, migrazione: null};
    }
    var stato = {schema: SCHEMA_STATO, pacchetto: {id: pkg.id, versione: pkg.versione, impronta: pkg.impronta},
                 revisore: typeof dati.revisore === "string" ? dati.revisore : "",
                 creato_il: dati.creato_il || ora(), aggiornato_il: dati.aggiornato_il || ora(),
                 unita: dati.unita, decontaminazione: dati.decontaminazione,
                 posizione: typeof dati.posizione === "string" ? dati.posizione : null,
                 esposizioni: formato === 2 ? dati.esposizioni : [], archivio: formato === 2 ? dati.archivio : [],
                 migrazioni: formato === 2 ? dati.migrazioni : []};
    var migrazione = null;
    if (formato === 1) {
      Object.keys(stato.unita).forEach(function (id) {
        stato.unita[id].prima_fissata = null;
        stato.unita[id].esposta_senza_primo_giudizio = null;
      });
    }
    var eff = applicaEsposizioni(ctx, stato);
    var riconciliazione = null;
    if (formato === 1) {
      migrazione = {da: SCHEMA_V1, a: SCHEMA_STATO, quando: ora(),
                    note: "soluzioni già mostrate trasformate in esposizioni dei loro record; unità collegate fissate " +
                          "o segnate come non indipendenti, anche se la loro soluzione era già stata mostrata",
                    esposizioni: stato.esposizioni.length, fissate: eff.fissate,
                    non_indipendenti: eff.esposte.concat(eff.riclassificate), bozze_archiviate: eff.archiviate};
      stato.migrazioni.push(migrazione);
    } else if (eff.riclassificate.length || eff.indipendenti.length || eff.provenienze.length) {
      /* stato del formato 2 scritto prima della correzione di 9ea9af1, che non riclassificava le unità già fissate */
      riconciliazione = {da: SCHEMA_STATO, a: SCHEMA_STATO, quando: ora(),
                         note: "indipendenza riclassificata dalle date: primo giudizio salvato rispetto alla prima " +
                               "esposizione, anche per le unità già fissate",
                         non_indipendenti: eff.riclassificate.concat(eff.esposte), indipendenti: eff.indipendenti,
                         provenienze_aggiornate: eff.provenienze};
      stato.migrazioni.push(riconciliazione);
    }
    return {errori: [], stato: stato, migrazione: migrazione, riconciliazione: riconciliazione};
  }

  /* ------------------------------------------------------------ unione di due stati */

  function senzaContatori(v) {
    var c = copia(v);
    delete c.tempo_ms;
    delete c.eventi;
    delete c.aggiornato_il;
    return canonico(c);
  }

  /* Tempo massimo, ultima modifica più recente, cronologia unita: gli eventi delle due versioni, senza doppioni, in
     ordine di tempo (a parità di tempo, prima quelli della versione corrente). */
  function unisciContatori(destinazione, a, b) {
    destinazione.tempo_ms = Math.max(a.tempo_ms, b.tempo_ms);
    var visti = {}, eventi = [];
    a.eventi.concat(b.eventi).forEach(function (e, i) {
      var k = canonico(e);
      if (!visti[k]) {
        visti[k] = true;
        eventi.push({e: e, i: i});
      }
    });
    destinazione.eventi = eventi.sort(function (x, y) {
      return x.e.quando < y.e.quando ? -1 : x.e.quando > y.e.quando ? 1 : x.i - y.i;
    }).map(function (x) { return copia(x.e); });
    destinazione.aggiornato_il = a.aggiornato_il > b.aggiornato_il ? a.aggiornato_il : b.aggiornato_il;
  }

  function primaDi(x, y) {  // la fissazione (o l'esposizione) più antica fra due, null se nessuna
    if (!x) {
      return y ? copia(y) : null;
    }
    if (!y) {
      return copia(x);
    }
    return copia(y.quando < x.quando ? y : x);
  }

  /* La versione non fissata aggiunge qualcosa rispetto a quella fissata? (un altro primo giudizio, una bozza diversa,
     un giudizio finale) */
  function aggiungeLavoro(alt, fis, unit) {
    var primaFis = canonico(fis.prima.varianti);
    return !!((alt.prima && canonico(alt.prima.varianti) !== primaFis)
              || (bozzaToccata(alt) && canonico(alt.bozza) !== primaFis)
              || (canonico(alt.dopo) !== canonico(dopoVuotoUnita(unit)) && canonico(alt.dopo) !== canonico(fis.dopo)));
  }

  /* Unisce lo stato importato in quello corrente e restituisce {stato, conflitti, risolti, ...}; non modifica gli
     ingressi. La conoscenza si unisce sempre: esposizioni (unione), primi giudizi fissati (vince la versione fissata;
     fra due fissate diverse quella fissata prima), soluzioni mostrate (la data più antica). Il lavoro in conflitto
     (stessa unità, nessuna delle due fissata, contenuti diversi; oppure stesso primo giudizio fissato e giudizio
     finale diverso) dipende da `modo`:
     - "senza_scelta": resta la versione corrente e il conflitto si segnala, perché scelga il revisore;
     - "browser": resta la versione corrente, quella del file va nell'archivio;
     - "file": vale la versione del file, quella corrente va nell'archivio;
     - "sostituisci": come "file", e in più le voci presenti solo nello stato corrente e non fissate vanno
       nell'archivio e si tolgono.
     Ogni versione scartata va nell'archivio. Alla fine le esposizioni si riapplicano a tutto lo stato. */
  function unisci(ctx, corrente, importato, modo) {
    var st = copia(corrente);
    var out = {stato: st, modo: modo, conflitti: [], risolti: [], nuove: 0, uguali: 0, solo_correnti: 0, tolte: 0,
               esposizioni_nuove: 0, archiviate: 0, fissate: [], non_indipendenti: []};
    importato.esposizioni.forEach(function (e) {
      if (aggiungiEsposizione(st, e)) {
        out.esposizioni_nuove += 1;
      }
    });
    importato.archivio.forEach(function (a) {
      var k = chiaveArchivio(a);
      if (!st.archivio.some(function (x) { return chiaveArchivio(x) === k; })) {
        st.archivio.push(copia(a));
      }
    });
    var archivioPrima = st.archivio.length;  // da qui contano le versioni scartate da questa unione
    importato.migrazioni.forEach(function (m) {
      if (!st.migrazioni.some(function (x) { return canonico(x) === canonico(m); })) {
        st.migrazioni.push(copia(m));
      }
    });
    var fileVince = modo === "file" || modo === "sostituisci";
    Object.keys(importato.unita).forEach(function (id) {
      var a = corrente.unita[id], b = importato.unita[id];
      var unit = ctx.idx.unita[id];
      if (!a) {
        st.unita[id] = copia(b);
        out.nuove += 1;
        return;
      }
      if (senzaContatori(a) === senzaContatori(b)) {
        unisciContatori(st.unita[id], a, b);
        out.uguali += 1;
        return;
      }
      var fa = !!a.prima_fissata, fb = !!b.prima_fissata;
      if (fa && fb && canonico(a.prima.varianti) !== canonico(b.prima.varianti)) {
        var tieniFile = b.prima_fissata.quando < a.prima_fissata.quando;
        var vince = tieniFile ? b : a, perde = tieniFile ? a : b;
        st.unita[id] = copia(vince);
        unisciContatori(st.unita[id], a, b);
        archivia(st, "unita", id, tieniFile ? "browser" : "file", "primo giudizio fissato diverso da quello fissato " +
                 "per primo (" + vince.prima_fissata.quando + "): non è il primo giudizio dell'unità", perde);
        out.risolti.push({id: id, regola: "due primi giudizi fissati diversi: vale quello fissato per primo",
                          tenuto: tieniFile ? "file" : "browser"});
        return;
      }
      if (fa && fb) {
        var m = copia(a);
        m.prima_fissata = primaDi(a.prima_fissata, b.prima_fissata);
        // indipendenza ricalcolata da applicaEsposizioni sulle esposizioni unite, con un evento se cambia
        m.soluzione_mostrata_il = !a.soluzione_mostrata_il ? b.soluzione_mostrata_il : !b.soluzione_mostrata_il ?
          a.soluzione_mostrata_il : (b.soluzione_mostrata_il < a.soluzione_mostrata_il ? b.soluzione_mostrata_il :
                                     a.soluzione_mostrata_il);
        if (canonico(a.dopo) !== canonico(b.dopo)) {
          if (fileVince) {
            m.dopo = copia(b.dopo);
            archivia(st, "unita", id, "browser", "giudizio finale sostituito da quello del file", a);
          } else if (modo === "browser") {
            archivia(st, "unita", id, "file", "giudizio finale del file non importato: vale quello del browser", b);
          }
          out.conflitti.push({id: id, sezione: "unita", parte: "giudizio finale", corrente: a.aggiornato_il,
                              file: b.aggiornato_il});
        }
        unisciContatori(m, a, b);
        st.unita[id] = m;
        return;
      }
      if (fa || fb) {
        var fis = fa ? a : b, alt = fa ? b : a;
        var nuova = copia(fis);
        // l'indipendenza non si copia dall'altra versione: la ricalcola applicaEsposizioni dalle esposizioni unite
        unisciContatori(nuova, a, b);
        st.unita[id] = nuova;
        if (aggiungeLavoro(alt, fis, unit)) {
          var dopoFis = alt.prima && alt.prima.salvata_il > fis.prima_fissata.quando;
          archivia(st, "unita", id, fa ? "file" : "browser", "versione non fissata, diversa dal primo giudizio fissato " +
                   "il " + fis.prima_fissata.quando + (alt.prima ? (dopoFis ? "; primo giudizio salvato dopo: non " +
                   "indipendente" : "; primo giudizio salvato prima della fissazione") : ""), alt);
          out.risolti.push({id: id, regola: "vale il primo giudizio fissato; l'altra versione è nell'archivio",
                            tenuto: fa ? "browser" : "file"});
        }
        return;
      }
      out.conflitti.push({id: id, sezione: "unita", parte: "primo giudizio", corrente: a.aggiornato_il,
                          file: b.aggiornato_il});
      if (fileVince) {
        st.unita[id] = copia(b);
        archivia(st, "unita", id, "browser", "sostituita dalla versione del file", a);
      } else if (modo === "browser") {
        archivia(st, "unita", id, "file", "versione del file non importata: vale quella del browser", b);
      }
    });
    Object.keys(corrente.unita).forEach(function (id) {
      if (importato.unita[id]) {
        return;
      }
      out.solo_correnti += 1;
      if (modo === "sostituisci" && !corrente.unita[id].prima_fissata) {
        archivia(st, "unita", id, "browser", "presente solo nel browser, tolta dalla sostituzione con il file",
                 corrente.unita[id]);
        delete st.unita[id];
        out.tolte += 1;
      }
    });
    Object.keys(importato.decontaminazione).forEach(function (id) {
      var a = corrente.decontaminazione[id], b = importato.decontaminazione[id];
      if (!a) {
        st.decontaminazione[id] = copia(b);
        out.nuove += 1;
      } else if (senzaContatori(a) === senzaContatori(b)) {
        unisciContatori(st.decontaminazione[id], a, b);
        out.uguali += 1;
      } else {
        out.conflitti.push({id: id, sezione: "decontaminazione", parte: "esame", corrente: a.aggiornato_il,
                            file: b.aggiornato_il});
        if (fileVince) {
          st.decontaminazione[id] = copia(b);
          archivia(st, "decontaminazione", id, "browser", "sostituita dalla versione del file", a);
        } else if (modo === "browser") {
          archivia(st, "decontaminazione", id, "file", "versione del file non importata: vale quella del browser", b);
        }
      }
    });
    Object.keys(corrente.decontaminazione).forEach(function (id) {
      if (importato.decontaminazione[id]) {
        return;
      }
      out.solo_correnti += 1;
      if (modo === "sostituisci") {
        archivia(st, "decontaminazione", id, "browser", "presente solo nel browser, tolta dalla sostituzione con il " +
                 "file", corrente.decontaminazione[id]);
        delete st.decontaminazione[id];
        out.tolte += 1;
      }
    });
    if (importato.revisore && (!st.revisore || modo === "sostituisci")) {
      st.revisore = importato.revisore;
    }
    var eff = applicaEsposizioni(ctx, st);
    out.fissate = eff.fissate;
    out.non_indipendenti = eff.esposte.concat(eff.riclassificate);
    out.indipendenti = eff.indipendenti;
    out.provenienze = eff.provenienze;
    out.archiviate = st.archivio.length - archivioPrima;
    return out;
  }

  /* Conoscenza contenuta in uno stato: per ogni unità fissata o esposta, la voce; e le esposizioni. Serve a capire se
     un'unione ha cambiato qualcosa di ciò che non deve mai andare perso. */
  function impronteConoscenza(stato) {
    var out = {esposizioni: stato.esposizioni.map(chiaveEsposizione).sort(), unita: {}};
    Object.keys(stato.unita).forEach(function (id) {
      var v = stato.unita[id];
      if (v.prima_fissata || v.esposta_senza_primo_giudizio || v.soluzione_mostrata_il) {
        out.unita[id] = canonico({prima: v.prima, prima_fissata: v.prima_fissata, soluzione: v.soluzione_mostrata_il,
                                  esposta: v.esposta_senza_primo_giudizio});
      }
    });
    return canonico(out);
  }

  root.Logica = {
    SCHEMA_STATO: SCHEMA_STATO, SCHEMA_V1: SCHEMA_V1, NON: NON, VALORI: VALORI, AIUTI: AIUTI, TIPI_DIFETTO: TIPI_DIFETTO,
    TSV_ESITI: TSV_ESITI, TSV_DECONTAMINAZIONE: TSV_DECONTAMINAZIONE, CAMPI_DOPO: CAMPI_DOPO, STILE: STILE,
    ammessi: ammessi, ora: ora, indice: indice, contesto: contesto, chiaveScena: chiaveScena,
    descriviCollegamento: descriviCollegamento, descriviFissazione: descriviFissazione,
    descriviEsposizione: descriviEsposizione, nuovoStato: nuovoStato, voce: voce, aggiornaBozza: aggiornaBozza,
    erroriPrima: erroriPrima, salvaPrima: salvaPrima, bozzaUgualeAPrima: bozzaUgualeAPrima, bloccanti: bloccanti,
    puoMostrare: puoMostrare, mostraSoluzione: mostraSoluzione, applicaEsposizioni: applicaEsposizioni,
    aggiornaDopo: aggiornaDopo, aggiungiTempo: aggiungiTempo, statoUnita: statoUnita, voceDecontam: voceDecontam,
    materialeMancante: materialeMancante, controlloDecontam: controlloDecontam, puoEscludere: puoEscludere,
    aggiornaEsame: aggiornaEsame, impostaEsitoDecontam: impostaEsitoDecontam, riepilogo: riepilogo,
    canonico: canonico, oggettoEsportato: oggettoEsportato, esportaJSON: esportaJSON, cella: cella,
    leggiCella: leggiCella, tsvEsiti: tsvEsiti, tsvDecontaminazione: tsvDecontaminazione,
    controllaImport: controllaImport, unisci: unisci, impronteConoscenza: impronteConoscenza, copia: copia
  };
})(window);
