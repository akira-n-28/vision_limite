/* Autotest di verifica.html: logica e interfaccia sui dati veri del pacchetto, senza toccare lo stato salvato nel
   browser (App.inizializza con persisti: false o con una memoria finta; scaricamenti e conferme finti). Senza DOM, cioè
   fuori da un browser, le parti che chiedono il DOM o le immagini risultano saltate, non superate; con il DOM simulato
   di scripts/review_package_dom.js (esecuzione con Duktape sul cluster) si salta solo il caricamento delle immagini.

   L'output non mostra mai soluzioni: nomi e dettagli contengono solo identificativi e conteggi, e alla fine un
   controllo cerca nell'output target, categorie, motivi, didascalie e valori delle schede (non presenti nel testo
   visibile della scheda, di almeno 4 caratteri); se ne trova, nasconde il dettaglio e fallisce. Deriva dall'autotest di
   revisioni/synth-v4-campione-2. */
(function (root) {
  "use strict";
  var L = root.Logica, P = root.PACCHETTO, S = root.SOLUZIONI, App = root.App;
  var righe = [];
  var conDom = typeof document !== "undefined";
  var simulato = !!(root.__DOM && root.__DOM.simulato);
  var tabella = conDom ? document.getElementById("risultati") : null;
  var rigaSegreti = null;
  root.RISULTATI_TEST = righe;

  function verifica(nome, cond, dettaglio) {
    righe.push({nome: nome, ok: !!cond, dettaglio: dettaglio === undefined ? "" : String(dettaglio)});
  }

  function salta(nome, motivo) {
    righe.push({nome: nome, ok: null, dettaglio: motivo});
  }

  function prova(nome, fn) {
    try {
      fn();
    } catch (e) {
      verifica(nome + ": eccezione", false, (e && e.stack) || String(e));
    }
  }

  /* ------------------------------------------------------------ nessuna soluzione nell'output */

  function confinato(testo, s) {  // s compare in testo come parola intera (o con bordi non alfanumerici)
    var parola = /[A-Za-z0-9_]/;
    var i = testo.indexOf(s);
    while (i >= 0) {
      var prima = i === 0 ? "" : testo.charAt(i - 1);
      var dopo = testo.charAt(i + s.length);
      if ((!parola.test(s.charAt(0)) || !parola.test(prima)) && (!parola.test(s.charAt(s.length - 1)) ||
                                                                  !parola.test(dopo))) {
        return true;
      }
      i = testo.indexOf(s, i + 1);
    }
    return false;
  }

  function visibile(s) {
    return [s.id, s.record, s.domanda, s.consegna].concat(s.opzioni).join("\n");
  }

  var SEGRETI = null;
  function segreti() {  // come secrets_card in scripts/mavis_review_package.py
    if (SEGRETI) {
      return SEGRETI;
    }
    var out = {};
    P.schede.forEach(function (s) {
      var x = S[s.id];
      if (!x) {
        return;
      }
      var cand = [x.target, x.sottocategoria, x.categoria, x.strato, x.motivo_selezione, x.annotazioni.didascalia];
      if (x.valore) {
        cand = cand.concat([x.valore.esatto, x.valore.visualizzato_previsto, x.valore.rappresentazione]);
      }
      if (x.c9) {
        cand = cand.concat([x.c9.motivo, x.c9.risposta]);
      }
      var vis = visibile(s);
      cand.forEach(function (t) {
        t = String(t === null || t === undefined ? "" : t).trim();
        if (t.length >= 4 && !confinato(vis, t)) {
          out[t] = true;
        }
      });
    });
    SEGRETI = Object.keys(out);
    return SEGRETI;
  }

  function trovaSegreti(testo) {
    return segreti().filter(function (s) { return confinato(testo, s); }).length;
  }

  function ripulisci() {
    if (!L || !S || !P) {
      return;
    }
    righe.forEach(function (r, i) {
      if (r !== rigaSegreti && !r.nascosto && trovaSegreti(r.nome + "\n" + r.dettaglio)) {
        r.nome = "riga " + (i + 1) + " dell'autotest";
        r.dettaglio = "[nome e dettaglio nascosti: contenevano un'informazione del secondo passaggio]";
        r.ok = false;
        r.nascosto = true;
      }
    });
    if (rigaSegreti) {
      righe.splice(righe.indexOf(rigaSegreti), 1);
    }
    var colpite = righe.filter(function (r) { return r.nascosto; }).map(function (r) { return r.nome; });
    rigaSegreti = {nome: "output dell'autotest senza target, categorie, motivi, didascalie o valori delle schede",
                   ok: !colpite.length, dettaglio: colpite.length ? "nascoste: " + colpite.join("; ") :
                     segreti().length + " stringhe del secondo passaggio cercate, nessuna trovata"};
    righe.push(rigaSegreti);
  }

  function disegna() {
    ripulisci();
    if (!conDom) {
      return;
    }
    tabella.innerHTML = "";
    var falliti = 0;
    righe.forEach(function (r) {
      var tr = document.createElement("tr");
      var a = document.createElement("td");
      a.textContent = r.ok === null ? "saltato" : r.ok ? "ok" : "FALLITO";
      a.className = r.ok === false ? "fail" : "ok";
      var b = document.createElement("td");
      b.textContent = r.nome;
      var c = document.createElement("td");
      c.textContent = r.dettaglio;
      tr.appendChild(a);
      tr.appendChild(b);
      tr.appendChild(c);
      tabella.appendChild(tr);
      falliti += r.ok === false ? 1 : 0;
    });
    document.getElementById("sintesi").textContent = (falliti ? falliti + " controlli FALLITI su " : "Tutti i ") +
      righe.length + (falliti ? "" : " controlli superati") + " · " + navigator.userAgent;
  }

  if (!L || !P || !S || !App) {
    verifica("dati e script caricati (dati/*.js, logica.js, app.js)", false, "manca qualcosa: estrai tutta la cartella");
    disegna();
    return;
  }
  verifica("dati e script caricati", true, P.id + " v" + P.versione + " · impronta " + P.impronta.slice(0, 16) +
           (simulato ? " · DOM simulato" : ""));
  var C = L.contesto(P, S);

  /* ------------------------------------------------------------ strumenti */

  function scheda(filtro) {
    return P.schede.filter(filtro)[0] || null;
  }

  function tipo(s) {
    var x = S[s.id];
    return x.categoria;
  }

  function riempiPrima(st, s, risposta) {
    L.aggiornaBozza(st, s, "risposta", risposta);
    L.aggiornaBozza(st, s, "leggibilita", "si");
    L.aggiornaBozza(st, s, "sufficienza", "si");
    return L.salvaPrima(st, s);
  }

  function riempiDopo(st, s) {
    L.modulo(C, s.id).scelte.forEach(function (c) {
      L.aggiornaDopo(C, st, s, c, c === "giudizio_finale" ? "nessun_problema" : "si");
    });
  }

  function rigaTsv(st, id) {
    var testa = L.tsvEsiti(C, st).split("\n")[0].split("\t");
    var r = (L.tsvEsiti(C, st).split("\n").filter(function (x) { return x.indexOf(id + "\t") === 0; })[0] || "")
      .split("\t");
    var o = {};
    testa.forEach(function (h, i) { o[h] = r[i]; });
    return o;
  }

  var VAL = scheda(function (s) { return tipo(s) === "lettura_valore"; });
  var REC = scheda(function (s) { return tipo(s) === "classificazione_recuperabile"; });
  var DIA = scheda(function (s) { return s.richiesta === "diagnostica"; });
  var ALT = scheda(function (s) { return tipo(s) === "altro_compito"; });
  var CTRL = scheda(function (s) { return s.controllo_noto; });

  /* ------------------------------------------------------------ 1. dati del pacchetto */
  prova("dati", function () {
    var ids = P.schede.map(function (s) { return s.id; });
    var attesi = ids.map(function (x, i) { return "M" + (i < 9 ? "0" : "") + (i + 1); });
    verifica("schede M01… nell'ordine, ciascuna con la sua soluzione (nascosta)", ids.join() === attesi.join() &&
             Object.keys(S).length === ids.length && ids.every(function (id) { return !!S[id]; }), ids.length);
    var rec = {}, gru = {}, img = {}, doppi = 0;
    P.schede.forEach(function (s) {
      doppi += (rec[s.record] ? 1 : 0) + (gru[s.gruppo] ? 1 : 0) + (img[s.immagine] ? 1 : 0);
      rec[s.record] = gru[s.gruppo] = img[s.immagine] = 1;
    });
    verifica("record, gruppi e immagini distinti (una scheda per gruppo)", doppi === 0, doppi);
    var nascoste = ["target", "risposta", "answer", "categoria", "sottocategoria", "strato", "cella", "regola",
                    "motivo", "motivo_selezione", "esito", "c9", "valore", "valori", "annotazioni", "didascalia",
                    "caption", "diagram_generation", "fatti", "confutazioni", "rappresentazione", "esatto",
                    "visualizzato_previsto", "risposta_lettura", "storica", "split", "stato", "compito",
                    "tipo_dichiarato", "elemento", "figura"];
    var fughe = [];
    (function cerca(x, dove) {
      if (Array.isArray(x)) {
        x.forEach(function (y, i) { cerca(y, dove + "[" + i + "]"); });
      } else if (x && typeof x === "object") {
        Object.keys(x).forEach(function (k) {
          if (nascoste.indexOf(k) >= 0) { fughe.push(dove + "." + k); }
          cerca(x[k], dove + "." + k);
        });
      }
    })(P.schede, "schede");
    verifica("primo passaggio: nessun target, annotazione, esito, categoria o motivo nei dati delle schede",
             !fughe.length, fughe.slice(0, 3).join(", "));
    var vis = P.schede.filter(function (s) {
      var x = S[s.id], v = visibile(s);
      return [x.annotazioni.didascalia, x.motivo_selezione, x.strato].some(function (t) { return confinato(v, t); });
    });
    verifica("didascalia, motivo e strato non compaiono nel testo visibile di nessuna scheda", !vis.length,
             vis.map(function (s) { return s.id; }).join(", "));
    var ctrl = P.schede.filter(function (s) { return s.controllo_noto; });
    verifica("un solo controllo noto, segnato nel primo passaggio", ctrl.length === 1, ctrl.length);
    var diag = P.schede.filter(function (s) { return s.richiesta === "diagnostica"; });
    verifica("richieste diagnostiche: senza target, dichiarate nel primo passaggio, con le opzioni della famiglia",
             diag.length > 0 && diag.every(function (s) {
               return S[s.id].target === null && !S[s.id].qa_candidata && s.opzioni.length === 3;
             }), diag.length);
    var opz = P.schede.filter(function (s) { return s.opzioni.length && s.richiesta === "qa"; });
    verifica("QA con opzioni: il target è una delle opzioni della domanda, e nessuna QA senza opzioni ne mostra",
             opz.every(function (s) { return s.opzioni.indexOf(S[s.id].target) >= 0; }) &&
             P.schede.every(function (s) { return !s.opzioni.length || S[s.id].compito === "shape_type"; }), opz.length);
    verifica("sono presenti letture di valori, classificazioni recuperabili e omesse e altri compiti",
             !!(VAL && REC && DIA && ALT && CTRL));
  });

  /* ------------------------------------------------------------ 2. nessun collegamento fra schede */
  prova("collegamenti", function () {
    var coll = P.schede.filter(function (s) { return C.collegati[s.id].length; });
    verifica("nessuna scheda collegata (record e gruppi distinti): ogni soluzione espone solo la sua scheda",
             !coll.length, coll.length);
  });

  /* ------------------------------------------------------------ 3. primo giudizio separato dalla soluzione */
  prova("primo giudizio", function () {
    var st = L.nuovoStato(P);
    verifica("senza primo giudizio la soluzione non si mostra", !L.puoMostrare(C, st, VAL).ok &&
             L.mostraSoluzione(C, st, VAL).length > 0 && !st.esposizioni.length);
    L.aggiornaBozza(st, VAL, "risposta", "x");
    verifica("primo giudizio incompleto (leggibilità e sufficienza mancanti): non si salva",
             L.salvaPrima(st, VAL).length === 2 && !st.schede[VAL.id].prima);
    L.aggiornaBozza(st, VAL, "risposta", "");
    L.aggiornaBozza(st, VAL, "non_determinabile", true);
    L.aggiornaBozza(st, VAL, "leggibilita", "no");
    L.aggiornaBozza(st, VAL, "sufficienza", "no");
    verifica("«non determinabile» vale come risposta", !L.salvaPrima(st, VAL).length && !!st.schede[VAL.id].prima);
    verifica("valori non ammessi rifiutati", L.aggiornaBozza(st, VAL, "leggibilita", "forse").length === 1);
    L.aggiornaBozza(st, VAL, "note", "modifica non salvata");
    verifica("modifiche non salvate: la soluzione non si mostra", !L.puoMostrare(C, st, VAL).ok);
    L.salvaPrima(st, VAL);
    verifica("il giudizio finale non si compila prima della soluzione",
             L.aggiornaDopo(C, st, VAL, "corretta_pixel", "si").length === 1);
    verifica("primo giudizio salvato: la soluzione si può mostrare", L.puoMostrare(C, st, VAL).ok);
    verifica("mostrata la soluzione: esposti record e gruppo della scheda", !L.mostraSoluzione(C, st, VAL).length &&
             st.esposizioni.length === 2 && st.esposizioni.every(function (e) { return e.tramite === VAL.id; }));
    verifica("dopo la soluzione il primo giudizio è fissato", !!st.schede[VAL.id].prima_fissata &&
             L.aggiornaBozza(st, VAL, "risposta", "ripensamento").length === 1 && L.salvaPrima(st, VAL).length === 1);
    verifica("lettura di un valore: scrittura visibile richiesta, prudenza del filtro non prevista",
             L.modulo(C, VAL.id).scelte.indexOf("scrittura_visibile") >= 0 &&
             L.aggiornaDopo(C, st, VAL, "prudenza_eccessiva", "si").length === 1 &&
             L.aggiornaDopo(C, st, VAL, "risposte_vere", "x").length === 1 &&
             !L.aggiornaDopo(C, st, VAL, "equivalenza", "uguale").length);
    verifica("giudizio finale incompleto: la scheda non è completa", L.statoScheda(C, VAL, st.schede[VAL.id]).fase ===
             "soluzione_vista" && L.mancantiDopo(C, VAL, st.schede[VAL.id]).length > 0);
    riempiDopo(st, VAL);
    L.aggiornaDopo(C, st, VAL, "giudizio_finale", "errore_grave");
    verifica("errore grave senza tipo di difetto: manca il tipo", L.mancantiDopo(C, VAL, st.schede[VAL.id])
             .indexOf("tipo_difetto") >= 0);
    L.aggiornaDopo(C, st, VAL, "tipo_difetto", "target_errato");
    verifica("giudizio finale completo: scheda completa e indipendente", L.statoScheda(C, VAL, st.schede[VAL.id]).fase ===
             "completo" && L.indipendente(C, VAL, st.schede[VAL.id]));
    var st2 = L.nuovoStato(P);
    riempiPrima(st2, DIA, "una o più classi");
    L.mostraSoluzione(C, st2, DIA);
    verifica("richiesta diagnostica: prudenza del filtro e classi vere previste, scrittura visibile no",
             L.modulo(C, DIA.id).scelte.indexOf("prudenza_eccessiva") >= 0 &&
             L.modulo(C, DIA.id).scelte.indexOf("scrittura_visibile") < 0 &&
             !L.aggiornaDopo(C, st2, DIA, "risposte_vere", "x").length &&
             L.aggiornaDopo(C, st2, DIA, "equivalenza", "x").length === 1);
    riempiPrima(st2, REC, "opzione");
    L.mostraSoluzione(C, st2, REC);
    verifica("classificazione recuperabile e altri compiti: né scrittura visibile né prudenza del filtro",
             [REC, ALT].every(function (s) {
               var m = L.modulo(C, s.id).scelte;
               return m.indexOf("scrittura_visibile") < 0 && m.indexOf("prudenza_eccessiva") < 0;
             }) && L.aggiornaDopo(C, st2, REC, "scrittura_visibile", "si").length === 1);
  });

  /* ------------------------------------------------------------ 4. controllo noto */
  prova("controllo noto", function () {
    var st = L.nuovoStato(P);
    riempiPrima(st, CTRL, "lettura");
    verifica("controllo noto: primo giudizio salvato ma non indipendente", !!st.schede[CTRL.id].prima &&
             !L.indipendente(C, CTRL, st.schede[CTRL.id]) && L.statoScheda(C, CTRL, st.schede[CTRL.id]).non_indipendente);
    verifica("controllo noto nel TSV: indipendente = no", rigaTsv(st, CTRL.id).indipendente === "no" &&
             rigaTsv(st, CTRL.id).controllo_noto === "si");
  });

  /* ------------------------------------------------------------ 5. collegamenti ed esposizioni (pacchetto di prova) */
  prova("esposizioni", function () {
    // due schede finte sullo stesso record e una sullo stesso gruppo: la logica dei collegamenti, che nel pacchetto vero
    // non scatta (record e gruppi distinti)
    var base = P.schede[0];
    function finta(id, record, gruppo) {
      return {id: id, posizione: 0, record: record, gruppo: gruppo, immagine: base.immagine,
              immagine_sha256: base.immagine_sha256, larghezza: 1, altezza: 1, richiesta: "qa", controllo_noto: false,
              domanda: "q", opzioni: [], consegna: "c"};
    }
    var pf = {id: "prova", versione: "1", impronta: "0000000000000000", schede: [finta("A", "r1", "g1"),
              finta("B", "r1", "g1"), finta("D", "r2", "g1"), finta("E", "r3", "g3")]};
    var sf = {A: S[base.id], B: S[base.id], D: S[base.id], E: S[base.id]};
    var cf = L.contesto(pf, sf), A = pf.schede[0], B = pf.schede[1], D = pf.schede[2], E = pf.schede[3];
    var st = L.nuovoStato(pf);
    function riempi(x, s) {
      L.aggiornaBozza(x, s, "risposta", "r");
      L.aggiornaBozza(x, s, "leggibilita", "si");
      L.aggiornaBozza(x, s, "sufficienza", "si");
      return L.salvaPrima(x, s);
    }
    verifica("collegate per record e per gruppo", cf.collegati.A.length === 2 && cf.collegati.D.length === 2 &&
             !cf.collegati.E.length);
    riempi(st, A);
    var p = L.puoMostrare(cf, st, A);
    verifica("schede collegate senza primo giudizio: la soluzione non si mostra", !p.ok && p.mancanti.length === 2);
    riempi(st, B);
    riempi(st, D);
    verifica("con il primo giudizio delle collegate la soluzione si mostra e le fissa",
             !L.mostraSoluzione(cf, st, A).length && !!st.schede.B.prima_fissata && !!st.schede.D.prima_fissata &&
             st.schede.B.prima_fissata.tramite === "A" && !st.schede.B.esposta_prima && L.indipendente(cf, B, st.schede.B));
    verifica("una scheda non collegata non è toccata dalla soluzione delle altre", !st.schede.E);
    var st3 = L.nuovoStato(pf);  // un'esposizione arrivata da un altro stato prima dei primi giudizi
    st3.esposizioni.push({chiave: "gruppo:g1", quando: L.ora(), tramite: "A"});
    L.applicaEsposizioni(cf, st3);
    verifica("una scheda esposta prima del suo primo giudizio è segnata e, salvato, il primo giudizio è fissato e non " +
             "indipendente", !!st3.schede.D.esposta_prima && !riempi(st3, D).length && !!st3.schede.D.prima_fissata &&
             !L.indipendente(cf, D, st3.schede.D) && !st3.schede.E);
    var vecchio = L.nuovoStato(pf);
    var u = L.unisci(cf, st, vecchio, "sostituisci");
    verifica("«sostituisci» con uno stato vuoto: esposizioni e primi giudizi fissati restano", u.stato.esposizioni.length ===
             st.esposizioni.length && !!u.stato.schede.A.soluzione_mostrata_il && !!u.stato.schede.B.prima_fissata);
    var w = L.unisci(cf, vecchio, st, "senza_scelta");
    verifica("unire lo stato con le esposizioni in uno vuoto le porta tutte", w.stato.esposizioni.length === 2 &&
             !!w.stato.schede.D.prima_fissata && w.nuove === 3);
  });

  /* ------------------------------------------------------------ 6. esportazione, importazione, unione */
  prova("esportazione", function () {
    var st = L.nuovoStato(P);
    riempiPrima(st, VAL, "valore\tcon tab\ne a capo \\ barra");
    L.mostraSoluzione(C, st, VAL);
    riempiDopo(st, VAL);
    riempiPrima(st, ALT, "altro");
    var testo = L.esportaJSON(C, st);
    var r = L.controllaImport(C, testo);
    verifica("il file di avanzamento si reimporta identico", !r.errori.length &&
             L.canonico(r.stato) === L.canonico(JSON.parse(testo)));
    var altro = JSON.parse(testo);
    altro.pacchetto.impronta = "f" + altro.pacchetto.impronta.slice(1);
    verifica("file di un altro pacchetto o versione: rifiutato", L.controllaImport(C, JSON.stringify(altro)).errori.length > 0);
    var ignota = JSON.parse(testo);
    ignota.schede.M99 = ignota.schede[VAL.id];
    verifica("scheda sconosciuta: rifiutato", L.controllaImport(C, JSON.stringify(ignota)).errori.length > 0);
    var valore = JSON.parse(testo);
    valore.schede[VAL.id].dopo.giudizio_finale = "corretto";
    verifica("valore non ammesso: rifiutato", L.controllaImport(C, JSON.stringify(valore)).errori.length > 0);
    var fuori = JSON.parse(testo);
    fuori.schede[ALT.id].dopo.scrittura_visibile = "si";
    verifica("campo compilato dove non è previsto: rifiutato", L.controllaImport(C, JSON.stringify(fuori)).errori.length > 0);
    verifica("JSON non valido: rifiutato", L.controllaImport(C, "{non valido").errori.length === 1);
    var tsv = L.tsvEsiti(C, st).split("\n");
    verifica("TSV: intestazione, una riga per scheda, celle con tab e a capo protetti",
             tsv[0] === L.TSV_ESITI.join("\t") && tsv.length === P.schede.length + 2 && tsv[P.schede.length + 1] === "" &&
             tsv.every(function (x) { return !x || x.split("\t").length === L.TSV_ESITI.length; }) &&
             rigaTsv(st, VAL.id).risposta_revisore === "valore\\tcon tab\\ne a capo \\\\ barra");
    var rv = rigaTsv(st, VAL.id), ra = rigaTsv(st, ALT.id);
    verifica("TSV: campi non previsti «non_applicabile», non compilati «non_compilato»",
             rv.prudenza_eccessiva === L.NA && rv.scrittura_visibile === "si" && ra.scrittura_visibile === L.NA &&
             ra.corretta_pixel === L.NON && ra.indipendente === "si" && rigaTsv(st, DIA.id).risposta_revisore === L.NON);
  });

  prova("unione", function () {
    var vecchio = L.nuovoStato(P);
    riempiPrima(vecchio, VAL, "vecchia");
    riempiPrima(vecchio, ALT, "file");
    var nuovo = L.controllaImport(C, L.esportaJSON(C, vecchio)).stato;
    L.mostraSoluzione(C, nuovo, VAL);
    L.aggiornaBozza(nuovo, ALT, "risposta", "browser");
    L.salvaPrima(nuovo, ALT);
    var u = L.unisci(C, nuovo, vecchio, "senza_scelta");
    verifica("importare un file vecchio: esposizioni e primi giudizi fissati restano",
             u.stato.esposizioni.length === 2 && !!u.stato.schede[VAL.id].soluzione_mostrata_il &&
             u.decise_da_fissazione.indexOf(VAL.id) >= 0);
    verifica("lavoro in conflitto senza scelta: resta quello del browser, l'altro nell'archivio",
             u.conflitti.length === 1 && u.conflitti[0] === ALT.id &&
             u.stato.schede[ALT.id].prima.bozza.risposta === "browser" &&
             u.stato.archivio.some(function (a) { return a.id === ALT.id && a.origine === "importato"; }));
    var f = L.unisci(C, nuovo, vecchio, "file");
    verifica("«usa il file»: vale il file nei conflitti, ma la soluzione mostrata resta",
             f.stato.schede[ALT.id].prima.bozza.risposta === "file" && !!f.stato.schede[VAL.id].soluzione_mostrata_il);
    var vuoto = L.nuovoStato(P);
    var s = L.unisci(C, nuovo, vuoto, "sostituisci");
    verifica("«sostituisci» con un file vuoto: tolte solo le schede senza primo giudizio fissato, nell'archivio",
             !!s.stato.schede[VAL.id] && !s.stato.schede[ALT.id] && s.stato.esposizioni.length === 2 &&
             s.stato.archivio.some(function (a) { return a.id === ALT.id; }));
    var again = L.unisci(C, u.stato, u.stato, "senza_scelta");
    verifica("unire uno stato con sé stesso non cambia la conoscenza", L.impronteConoscenza(again.stato) ===
             L.impronteConoscenza(u.stato) && !again.conflitti.length && !again.nuove);
  });

  /* ------------------------------------------------------------ 7. interfaccia (DOM) */
  var scaricati = [];
  function opzioniProva(mem) {
    return {persisti: !!mem, memoria: mem || null, conferma: function () { return true; },
            scarica: function (nome, testo) { scaricati.push({nome: nome, byte: testo.length}); }};
  }
  function memoriaFinta() {
    var dati = {};
    return {dati: dati, getItem: function (k) { return Object.prototype.hasOwnProperty.call(dati, k) ? dati[k] : null; },
            setItem: function (k, v) { dati[k] = String(v); }};
  }
  var chiave = "revisione:" + P.id + ":v" + P.versione + ":" + P.impronta.slice(0, 16);

  if (!conDom) {
    salta("interfaccia (DOM), ripresa nel browser e immagini", "fuori dal browser: si provano aprendo verifica.html");
    ripulisci();
    return;
  }
  prova("interfaccia", function () {
    var st = L.nuovoStato(P);
    App.inizializza(P, S, st, opzioniProva(null));
    var u = scheda(function (s) {
      var t = S[s.id].target;
      return t && t.length >= 4 && s.richiesta === "qa" && !s.opzioni.length && visibile(s).indexOf(t) < 0;
    });
    var nodo = App.costruisciVistaScheda(u);
    verifica("prima del giudizio: nessuna soluzione, annotazione o categoria nella pagina",
             !nodo.querySelector(".soluzione") && !confinato(nodo.textContent, S[u.id].target) &&
             !confinato(nodo.textContent, S[u.id].annotazioni.didascalia) &&
             !confinato(nodo.textContent, S[u.id].sottocategoria) && !confinato(nodo.textContent, S[u.id].categoria));
    // nessun identificativo nei dettagli: la scheda di prova è scelta per un target lungo, e il suo nome lo direbbe
    verifica("il controllo dell'output riconosce un target reale (prova interna, il target non si stampa)",
             trovaSegreti(u.id + " · target " + S[u.id].target) > 0 && trovaSegreti(u.id) === 0);
    verifica("la domanda è esatta", nodo.querySelector("pre.domanda").textContent === u.domanda);
    var conOpz = App.costruisciVistaScheda(REC), senza = App.costruisciVistaScheda(VAL);
    verifica("opzioni mostrate solo se appartengono alla domanda", conOpz.querySelectorAll("ul.opzioni li").length ===
             REC.opzioni.length && !senza.querySelector("ul.opzioni"));
    var d = App.costruisciVistaScheda(DIA), k = App.costruisciVistaScheda(CTRL);
    verifica("richiesta diagnostica dichiarata solo sulle schede diagnostiche",
             !!d.querySelector("p.diagnostica") && !conOpz.querySelector("p.diagnostica") &&
             !senza.querySelector("p.diagnostica"));
    verifica("controllo noto segnato solo sulla sua scheda", !!k.querySelector("p.controllo") &&
             !conOpz.querySelector("p.controllo"));
    var titoli = P.schede.map(function (s) { return App.costruisciVistaScheda(s).querySelector("h2").textContent; });
    verifica("titoli delle schede neutri (solo l'identificativo)", titoli.every(function (t, i) {
      return t === "Scheda " + P.schede[i].id; }));
    L.aggiornaBozza(st, u, "risposta", "prova");
    L.aggiornaBozza(st, u, "leggibilita", "si");
    nodo = App.costruisciVistaScheda(u);
    nodo.querySelector("button.salva-prima").click();
    verifica("salvataggio incompleto dal pulsante: errore e nessun primo giudizio", !st.schede[u.id].prima);
    L.aggiornaBozza(st, u, "sufficienza", "si");
    nodo = App.costruisciVistaScheda(u);
    nodo.querySelector("button.salva-prima").click();
    nodo = App.costruisciVistaScheda(u);
    var bottone = nodo.querySelector("button.rivela");
    verifica("primo giudizio salvato dal pulsante: la soluzione si può mostrare", !!st.schede[u.id].prima && !!bottone &&
             !bottone.disabled);
    bottone.click();
    nodo = App.costruisciVistaScheda(u);
    verifica("dopo il clic: soluzione, target e annotazioni visibili, primo giudizio bloccato",
             !!App.stato().schede[u.id].soluzione_mostrata_il && !!nodo.querySelector(".soluzione") &&
             nodo.textContent.indexOf(S[u.id].target) >= 0 &&
             Array.prototype.every.call(nodo.querySelectorAll(".prima textarea, .prima select, .prima input"),
                                        function (x) { return x.disabled; }) && !nodo.querySelector("button.salva-prima"));
    var campi = Array.prototype.map.call(nodo.querySelectorAll(".dopo select"), function (x) {
      return x.getAttribute("data-campo"); });
    verifica("giudizio finale: i campi previsti per la scheda", campi.join() === L.modulo(C, u.id).scelte.join());
    var tutte = P.schede.every(function (s) {
      var w = L.nuovoStato(P);
      App.inizializza(P, S, w, opzioniProva(null));
      L.aggiornaBozza(w, s, "non_determinabile", true);
      L.aggiornaBozza(w, s, "leggibilita", "no");
      L.aggiornaBozza(w, s, "sufficienza", "no");
      L.salvaPrima(w, s);
      L.mostraSoluzione(C, w, s);
      var n = App.costruisciVistaScheda(s);
      var sel = Array.prototype.map.call(n.querySelectorAll(".dopo select"), function (x) {
        return x.getAttribute("data-campo"); });
      return !!n.querySelector(".soluzione") && sel.join() === L.modulo(C, s.id).scelte.join() &&
        !!n.querySelector("[data-campo=equivalenza]") === L.modulo(C, s.id).lettura &&
        !!n.querySelector("[data-campo=risposte_vere]") === L.modulo(C, s.id).diagnostica;
    });
    verifica("secondo passaggio costruito per tutte le schede, coi campi previsti per ciascuna", tutte, P.schede.length);
    App.inizializza(P, S, st, opzioniProva(null));
    verifica("riepilogo costruito, con il registro dei record esposti",
             !!App.costruisciRiepilogo().querySelector("table.riepilogo") &&
             !!App.costruisciRiepilogo().querySelector("table.esposizioni"));
  });

  prova("interfaccia: importazione", function () {
    var st = L.nuovoStato(P);
    riempiPrima(st, VAL, "uno");
    riempiPrima(st, ALT, "browser");
    var vecchio = JSON.parse(L.esportaJSON(C, st));
    vecchio.schede[ALT.id].prima.bozza.risposta = "file vecchio";
    vecchio.schede[ALT.id].bozza.risposta = "file vecchio";
    L.mostraSoluzione(C, st, VAL);
    scaricati = [];
    App.inizializza(P, S, st, opzioniProva(null));
    var msg = document.createElement("p"), box = document.createElement("div");
    var u = App.importaTesto(JSON.stringify(vecchio), msg, box);
    var dopo = App.stato();
    verifica("importare un file vecchio: copia del browser scaricata prima, esposizioni e blocchi restano",
             !!u && scaricati.length === 1 && scaricati[0].nome.indexOf("_copia_del_browser_") > 0 &&
             dopo.esposizioni.length === 2 && !!dopo.schede[VAL.id].soluzione_mostrata_il);
    verifica("importare un file vecchio: il lavoro in conflitto non si sovrascrive senza una scelta",
             u.conflitti.length === 1 && dopo.schede[ALT.id].prima.bozza.risposta === "browser" &&
             !!box.querySelector("button.sostituisci"));
    box.querySelector("button.usa-file").click();
    var fine = App.stato();
    verifica("«Usa le versioni del file»: vale il file nei conflitti, la soluzione mostrata resta",
             fine.schede[ALT.id].prima.bozza.risposta === "file vecchio" && !!fine.schede[VAL.id].soluzione_mostrata_il &&
             fine.archivio.some(function (a) { return a.id === ALT.id; }));
    var rifiutato = App.importaTesto("{non valido", msg, box);
    verifica("file non valido: rifiutato con un messaggio, stato invariato", rifiutato === null &&
             msg.className.indexOf("errore") >= 0 && !!App.stato().schede[VAL.id].soluzione_mostrata_il);
  });

  prova("interfaccia: ripresa nel browser", function () {
    var vecchio = L.nuovoStato(P);
    riempiPrima(vecchio, VAL, "uno");
    var nuovo = L.controllaImport(C, L.esportaJSON(C, vecchio)).stato;
    L.mostraSoluzione(C, nuovo, VAL);
    var mem = memoriaFinta();
    mem.setItem(chiave, L.esportaJSON(C, nuovo));
    App.inizializza(P, S, null, opzioniProva(mem));
    verifica("ripresa: lo stato salvato nel browser si ricarica con esposizioni e blocchi",
             App.stato().esposizioni.length === 2 && !!App.stato().schede[VAL.id].prima_fissata &&
             App.esito().indexOf("Ripreso") === 0);
    mem.setItem(chiave, L.esportaJSON(C, vecchio));  // un'altra scheda rimasta indietro scrive uno stato vecchio
    App.salvaOra();
    var salvato = L.controllaImport(C, mem.getItem(chiave)).stato;
    verifica("ripresa: uno stato vecchio scritto da un'altra scheda non cancella l'esposizione al salvataggio",
             salvato.esposizioni.length === 2 && !!salvato.schede[VAL.id].soluzione_mostrata_il);
    var mem2 = memoriaFinta();
    mem2.setItem(chiave, L.esportaJSON(C, vecchio));
    App.inizializza(P, S, null, opzioniProva(mem2));
    mem2.setItem(chiave, L.esportaJSON(C, nuovo));  // un'altra scheda mostra la soluzione
    App.salvaOra();
    verifica("ripresa: un'esposizione scritta da un'altra scheda si unisce allo stato in memoria",
             App.stato().esposizioni.length === 2 && !!App.stato().schede[VAL.id].prima_fissata);
    var mem3 = memoriaFinta();
    mem3.setItem(chiave, "{non valido");
    App.inizializza(P, S, null, opzioniProva(mem3));
    var esito3 = App.esito();
    App.salvaOra();
    var riserve = Object.keys(mem3.dati).filter(function (x) { return x.indexOf(chiave + ":riserva:non-caricato:") === 0; });
    verifica("ripresa: uno stato salvato illeggibile non si perde (copia di riserva) e lo si segnala",
             riserve.length === 1 && mem3.dati[riserve[0]] === "{non valido" && esito3.indexOf("non è valido") >= 0);
    App.inizializza(P, S, L.nuovoStato(P), opzioniProva(null));
  });

  disegna();

  /* ------------------------------------------------------------ 8. immagini (asincrono) */
  if (simulato) {
    salta("immagini: tutte caricate dalla cartella", "DOM simulato: le immagini si caricano solo in un browser; " +
          "presenza e sha256 sono controllati da mavis_review_package.py verifica");
    disegna();
    return;
  }
  var elenco = P.schede.map(function (s) { return s.immagine; }), caricate = 0, errori = [], piccole = [];
  var riga = {nome: "immagini: tutte caricate dalla cartella, alla risoluzione registrata", ok: false,
              dettaglio: "in corso…"};
  righe.push(riga);
  disegna();
  P.schede.forEach(function (s) {
    var img = new Image();
    img.onload = function () {
      caricate += 1;
      if (img.naturalWidth !== s.larghezza || img.naturalHeight !== s.altezza) {
        piccole.push(s.id);
      }
      fine();
    };
    img.onerror = function () { errori.push(s.immagine); fine(); };
    img.src = s.immagine;
  });
  function fine() {
    if (caricate + errori.length < elenco.length) {
      return;
    }
    riga.ok = !errori.length && !piccole.length && caricate === elenco.length;
    riga.dettaglio = caricate + " su " + elenco.length + (errori.length ? "; non trovate: " + errori.slice(0, 5).join(", ") :
      "") + (piccole.length ? "; dimensioni diverse: " + piccole.slice(0, 5).join(", ") : "");
    disegna();
  }
})(window);
