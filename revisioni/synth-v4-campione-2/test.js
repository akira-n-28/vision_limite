/* Autotest di verifica.html: logica e interfaccia sui dati veri del pacchetto, senza toccare lo stato salvato nel
   browser (App.inizializza con persisti: false; nessun uso di localStorage). */
(function (root) {
  "use strict";
  var L = root.Logica, P = root.PACCHETTO, S = root.SOLUZIONI, E = root.EVIDENZE, App = root.App;
  var righe = [];
  var tabella = document.getElementById("risultati");

  function verifica(nome, cond, dettaglio) {
    righe.push({nome: nome, ok: !!cond, dettaglio: dettaglio === undefined ? "" : String(dettaglio)});
  }

  function prova(nome, fn) {
    try {
      fn();
    } catch (e) {
      verifica(nome + ": eccezione", false, (e && e.stack) || String(e));
    }
  }

  function disegna() {
    tabella.innerHTML = "";
    var falliti = 0;
    righe.forEach(function (r) {
      var tr = document.createElement("tr");
      var a = document.createElement("td");
      a.textContent = r.ok ? "ok" : "FALLITO";
      a.className = r.ok ? "ok" : "fail";
      var b = document.createElement("td");
      b.textContent = r.nome;
      var c = document.createElement("td");
      c.textContent = r.dettaglio;
      tr.appendChild(a);
      tr.appendChild(b);
      tr.appendChild(c);
      tabella.appendChild(tr);
      falliti += r.ok ? 0 : 1;
    });
    document.getElementById("sintesi").textContent = (falliti ? falliti + " controlli FALLITI su " : "Tutti i ") +
      righe.length + (falliti ? "" : " controlli superati") + " · " + navigator.userAgent;
  }

  if (!L || !P || !S || !E || !App) {
    verifica("dati e script caricati (dati/*.js, logica.js, app.js)", false, "manca qualcosa: estrai tutta la cartella");
    disegna();
    return;
  }
  verifica("dati e script caricati", true, P.id + " v" + P.versione + " · impronta " + P.impronta.slice(0, 16));

  function unita(sezione, filtro) {
    return P[sezione].filter(filtro)[0];
  }

  function riempiPrima(st, u, risposta) {
    u.varianti.forEach(function (k) {
      L.aggiornaBozza(st, u, k, "risposta", risposta + (k ? "-" + k : ""));
      L.aggiornaBozza(st, u, k, "leggibilita", "si");
    });
    return L.salvaPrima(st, u);
  }

  function riempiDopo(st, u) {
    u.varianti.forEach(function (k) {
      ["corretta", "coerente_con_annotazioni", "rispondibile_da_immagine"].forEach(function (c) {
        L.aggiornaDopo(st, u, k, c, "si");
      });
      L.aggiornaDopo(st, u, k, "gravita", "nessun_difetto");
    });
    if (u.coppia) {
      L.aggiornaDopo(st, u, "coppia", "contratto_coppia", "si");
      L.aggiornaDopo(st, u, "coppia", "gravita", "nessun_difetto");
    }
  }

  /* ------------------------------------------------------------ 1. dati del pacchetto */
  prova("dati", function () {
    var nCorr = P.decontaminazione.reduce(function (s, d) { return s + d.elenco.length; }, 0);
    verifica("380 unità casuali, 175 casi mirati, 264 candidati, 755 corrispondenze",
             P.campione.length === 380 && P.mirati.length === 175 && P.decontaminazione.length === 264 && nCorr === 755,
             [P.campione.length, P.mirati.length, P.decontaminazione.length, nCorr].join(", "));
    verifica("righe dei moduli: 655 in esiti, 264 in esiti_decontaminazione",
             P.righe_esiti.length === 655 && P.righe_esiti_decontaminazione.length === 264);
    var perStrato = {};
    P.campione.forEach(function (u) { perStrato[u.gruppo] = (perStrato[u.gruppo] || 0) + 1; });
    verifica("unità per strato uguali al disegno", Object.keys(P.strati).every(function (s) {
      return perStrato[s] === P.strati[s].n; }), JSON.stringify(perStrato));
    var ids = {}, doppi = 0;
    P.campione.concat(P.mirati, P.decontaminazione).forEach(function (x) { doppi += ids[x.id] ? 1 : 0; ids[x.id] = 1; });
    verifica("identificativi unici", doppi === 0, doppi);
    var tutte = P.campione.concat(P.mirati);
    verifica("ogni unità ha le sue soluzioni (nascoste) e ogni record il suo materiale", tutte.every(function (u) {
      return S[u.id] && u.immagini.every(function (im) { return E[im.record]; });
    }));
    var nonLeggibili = Object.keys(E).filter(function (k) {
      try { return !JSON.parse(E[k].json).qa; } catch (e) { return true; }
    });
    verifica("JSON dei record leggibili", !nonLeggibili.length, nonLeggibili.slice(0, 3).join(", "));
    var senzaRiga = tutte.filter(function (u) {
      return u.origine_prompt === "loader" && u.domande_nel_prompt.length > 1
        && u.prompt.split("\n").filter(function (r) { return r.indexOf(u.numero_nel_prompt + ". ") === 0; }).length !== 1;
    });
    verifica("prompt con più domande: la riga della domanda campionata si ritrova una volta", !senzaRiga.length,
             senzaRiga.slice(0, 3).map(function (u) { return u.id; }).join(", "));
    var senzaDomanda = tutte.filter(function (u) { return u.livello === "qa" && u.prompt.indexOf(u.domanda) < 0; });
    verifica("la domanda campionata è nel prompt", !senzaDomanda.length, senzaDomanda.length);
    var nascoste = ["risposta", "answer", "opzioni", "options", "esito_checker", "motivo", "misura",
                    "stato_scena_sha256", "questions"];
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
    })(tutte, "unita");
    verifica("primo passaggio: nessun target, opzione, esito del checker o motivo nei dati delle unità", !fughe.length,
             fughe.slice(0, 3).join(", "));
    var coppie = tutte.filter(function (u) { return u.coppia; });
    verifica("50 coppie con varianti a e b e soluzioni per entrambe", coppie.length === 50 && coppie.every(function (u) {
      return u.varianti.join() === "a,b" && S[u.id].varianti.map(function (s) { return s.variante; }).join() === "a,b";
    }), coppie.length);
  });

  /* ------------------------------------------------------------ 2. primo giudizio separato dalla soluzione */
  prova("primo giudizio", function () {
    var u = unita("campione", function (x) { return x.livello === "qa" && !x.coppia; });
    var st = L.nuovoStato(P);
    verifica("la soluzione non si mostra senza un primo giudizio salvato", L.mostraSoluzione(st, u).length > 0);
    verifica("il primo giudizio vuoto non si salva", L.salvaPrima(st, u).length > 0 && L.voce(st, u).prima === null);
    L.aggiornaBozza(st, u, "", "risposta", "right");
    L.aggiornaBozza(st, u, "", "leggibilita", "si");
    verifica("il primo giudizio completo si salva", L.salvaPrima(st, u).length === 0 && L.voce(st, u).prima !== null);
    L.aggiornaBozza(st, u, "", "risposta", "left");
    verifica("con modifiche non salvate la soluzione resta nascosta", !L.puoMostrare(st, u).ok);
    L.aggiornaBozza(st, u, "", "risposta", "right");
    verifica("dopo il salvataggio la soluzione si mostra", L.mostraSoluzione(st, u).length === 0
             && !!L.voce(st, u).soluzione_mostrata_il);
    verifica("dopo la soluzione il primo giudizio non si modifica",
             L.aggiornaBozza(st, u, "", "risposta", "up").length > 0 && L.salvaPrima(st, u).length > 0
             && L.voce(st, u).prima.varianti[""].risposta === "right");
    verifica("dopo la soluzione: fase soluzione_vista finché il giudizio finale è incompleto",
             L.statoUnita(u, L.voce(st, u)).fase === "soluzione_vista");
    L.aggiornaDopo(st, u, "", "gravita", "grave");
    verifica("gravità grave senza tipo di difetto: incompleto", L.statoUnita(u, L.voce(st, u)).mancanti.join()
             .indexOf("tipo_difetto") >= 0);
    riempiDopo(st, u);
    verifica("giudizio finale compilato: completo", L.statoUnita(u, L.voce(st, u)).fase === "completo");
    L.aggiornaDopo(st, u, "", "corretta", "incerto");
    verifica("un valore incerto si segnala", L.statoUnita(u, L.voce(st, u)).incerto);
    var altra = P.campione[1];
    verifica("il giudizio finale non si compila prima della soluzione",
             L.aggiornaDopo(st, altra, altra.varianti[0], "corretta", "si").length > 0);
    var v = P.campione[2];
    L.aggiornaBozza(st, v, v.varianti[0], "non_riesco", true);
    L.aggiornaBozza(st, v, v.varianti[0], "leggibilita", "no");
    verifica("\"non riesco a rispondere\" vale come risposta", L.salvaPrima(st, v).length === 0);
    var riga = L.tsvEsiti(P, st).split("\n").filter(function (r) { return r.indexOf(v.id + "\t") === 0; })[0] || "";
    verifica("nel TSV: non_riesco_a_rispondere", riga.split("\t")[3] === "non_riesco_a_rispondere", riga.slice(0, 120));
  });

  /* ------------------------------------------------------------ 3. coppie */
  prova("coppie", function () {
    var u = unita("campione", function (x) { return x.coppia; });
    var st = L.nuovoStato(P);
    L.aggiornaBozza(st, u, "a", "risposta", "9");
    L.aggiornaBozza(st, u, "a", "leggibilita", "si");
    verifica("coppia: il primo giudizio chiede entrambe le varianti", L.salvaPrima(st, u).length > 0);
    L.aggiornaBozza(st, u, "b", "risposta", "11");
    L.aggiornaBozza(st, u, "b", "leggibilita", "incerto");
    verifica("coppia: con a e b il primo giudizio si salva", L.salvaPrima(st, u).length === 0);
    L.mostraSoluzione(st, u);
    riempiDopo(st, u);
    L.aggiornaDopo(st, u, "b", "corretta", "no");
    L.aggiornaDopo(st, u, "coppia", "contratto_coppia", "no");
    var v = L.voce(st, u);
    verifica("coppia: risposte e giudizi distinti per a e b", v.prima.varianti.a.risposta === "9"
             && v.prima.varianti.b.risposta === "11" && v.dopo.varianti.a.corretta === "si"
             && v.dopo.varianti.b.corretta === "no");
    var righe = L.tsvEsiti(P, st).split("\n").filter(function (r) { return r.indexOf(u.id + "\t") === 0; })
      .map(function (r) { return r.split("\t"); });
    verifica("coppia: righe a, b e coppia nel TSV, con risposte e contratto",
             righe.length === 3 && righe[0][1] === "a" && righe[0][3] === "9" && righe[1][1] === "b" && righe[1][3] === "11"
             && righe[1][4] === "no" && righe[2][1] === "coppia" && righe[2][8] === "no" && righe[0][8] === "non_applicabile",
             JSON.stringify(righe.map(function (r) { return r.slice(1, 9); })));
    verifica("coppia: completa con i giudizi di a, b e del contratto", L.statoUnita(u, v).fase === "completo");
  });

  /* ------------------------------------------------------------ 4. casi sulla scena intera */
  prova("scene", function () {
    var u = unita("mirati", function (x) { return x.livello === "scena"; });
    var st = L.nuovoStato(P);
    L.aggiornaBozza(st, u, "", "leggibilita", "si");
    verifica("caso sulla scena intera: basta il giudizio iniziale", L.salvaPrima(st, u).length === 0, u.id);
  });

  /* ------------------------------------------------------------ 5. esportazione e importazione */
  var speciale = "√2 · x² ≤ 3 «virgolette» \"doppie\" 'singole'\tTAB\nseconda riga\r\nterza\\barra 😀 è à ∠ABC = 90°";
  prova("esportazione", function () {
    var st = L.nuovoStato(P);
    var u = unita("campione", function (x) { return x.livello === "qa" && !x.coppia; });
    L.aggiornaBozza(st, u, "", "risposta", speciale);
    L.aggiornaBozza(st, u, "", "note", speciale);
    L.aggiornaBozza(st, u, "", "leggibilita", "si");
    L.salvaPrima(st, u);
    L.mostraSoluzione(st, u);
    L.aggiornaDopo(st, u, "", "note", speciale);
    var c = unita("campione", function (x) { return x.coppia; });
    riempiPrima(st, c, "x");
    var d = P.decontaminazione[0];
    L.aggiornaEsame(st, d, d.elenco[0].id, "note", speciale);
    L.impostaEsitoDecontam(st, d, "note", speciale);
    st.revisore = "Revisore è";
    var testo = L.esportaJSON(P, st);
    var r = L.controllaImport(P, testo);
    verifica("il file esportato si reimporta senza errori", !r.errori.length, r.errori.join("; "));
    verifica("reimportato identico: risposte, note su più righe, caratteri speciali, coppie, decontaminazione",
             r.stato && L.canonico(r.stato.unita) === L.canonico(st.unita)
             && L.canonico(r.stato.decontaminazione) === L.canonico(st.decontaminazione)
             && r.stato.unita[u.id].prima.varianti[""].risposta === speciale && r.stato.revisore === "Revisore è");
    verifica("celle TSV: niente tabulazioni o a capo, e si rileggono uguali",
             !/[\t\n\r]/.test(L.cella(speciale)) && L.leggiCella(L.cella(speciale)) === speciale.replace(/\r\n/g, "\n"));
    var tsv = L.tsvEsiti(P, st).split("\n");
    verifica("esiti.tsv: intestazione dei moduli e 655 righe da 13 colonne",
             tsv[0] === L.TSV_ESITI.join("\t") && tsv.length === 657 && tsv[656] === ""
             && tsv.slice(1, 656).every(function (x) { return x.split("\t").length === 13; }));
    var riga = tsv.filter(function (x) { return x.indexOf(u.id + "\t") === 0; })[0].split("\t");
    verifica("esiti.tsv: risposta e note con i caratteri speciali conservati",
             L.leggiCella(riga[3]) === speciale.replace(/\r\n/g, "\n") && L.leggiCella(riga[12]) === speciale.replace(/\r\n/g, "\n"));
    var non = tsv.filter(function (x) { return x.indexOf(P.campione[5].id + "\t") === 0; })[0].split("\t");
    verifica("esiti.tsv: un'unità non toccata è non_compilato, non approvata", non[3] === "non_compilato"
             && non[4] === "non_compilato" && non[9] === "non_compilato");
    var dt = L.tsvDecontaminazione(P, st).split("\n");
    verifica("esiti_decontaminazione.tsv: intestazione e 264 righe da 6 colonne",
             dt[0] === L.TSV_DECONTAMINAZIONE.join("\t") && dt.length === 266 && dt.slice(1, 265).every(function (x) {
               return x.split("\t").length === 6; }));
  });

  prova("importazione rifiutata", function () {
    var st = L.nuovoStato(P);
    var u = P.campione[0];
    riempiPrima(st, u, "a");
    var buono = JSON.parse(L.esportaJSON(P, st));
    function rifiuta(nome, modifica) {
      var x = JSON.parse(JSON.stringify(buono));
      modifica(x);
      verifica("importazione rifiutata: " + nome, L.controllaImport(P, JSON.stringify(x)).errori.length > 0);
    }
    rifiuta("impronta di un altro pacchetto", function (x) { x.pacchetto.impronta = "0".repeat(64); });
    rifiuta("versione diversa", function (x) { x.pacchetto.versione = "0"; });
    rifiuta("schema diverso", function (x) { x.schema = "altro/1"; });
    rifiuta("unità sconosciuta", function (x) { x.unita["C-coord-999"] = x.unita[u.id]; });
    rifiuta("record diverso", function (x) { x.unita[u.id].varianti[u.varianti[0]].record = "synth-0000000"; });
    rifiuta("valore non ammesso", function (x) { x.unita[u.id].dopo.varianti[u.varianti[0]].corretta = "forse"; });
    rifiuta("soluzione mostrata senza primo giudizio", function (x) {
      x.unita[u.id].prima = null;
      x.unita[u.id].soluzione_mostrata_il = "2026-11-05T10:00:00.000Z";
    });
    verifica("un file non JSON è rifiutato", L.controllaImport(P, "non è json").errori.length > 0);
  });

  prova("conflitti", function () {
    var u = P.campione[0], w = P.campione[1];
    var a = L.nuovoStato(P), b = L.nuovoStato(P);
    riempiPrima(a, u, "uno");
    riempiPrima(b, u, "due");
    riempiPrima(b, w, "tre");
    var c = L.confronta(a, b);
    verifica("conflitto: risposte diverse per la stessa unità non si uniscono", c.conflitti.length === 1
             && c.conflitti[0].id === u.id && c.unito.unita[u.id].prima.varianti[u.varianti[0]].risposta.indexOf("uno") === 0
             && a.unita[u.id].prima.varianti[u.varianti[0]].risposta.indexOf("uno") === 0, JSON.stringify(c.conflitti));
    var a2 = L.copia(a);
    a2.unita[u.id].tempo_ms += 5000;
    var c2 = L.confronta(a, a2);
    verifica("solo il tempo diverso: nessun conflitto, si tiene il tempo maggiore", !c2.conflitti.length
             && c2.unito.unita[u.id].tempo_ms === a2.unita[u.id].tempo_ms);
    var c3 = L.confronta(L.nuovoStato(P), b);
    verifica("stato vuoto: il file si importa per intero", !c3.conflitti.length && c3.nuove === 2);
  });

  /* ------------------------------------------------------------ 6. decontaminazione */
  prova("decontaminazione", function () {
    var d = P.decontaminazione.filter(function (x) { return x.elenco.length >= 2 && !L.materialeMancante(x).length; })[0];
    var st = L.nuovoStato(P);
    verifica("«no» rifiutato con corrispondenze non esaminate", L.impostaEsitoDecontam(st, d, "esito", "no").length > 0
             && L.voceDecontam(st, d).esito === L.NON, d.id);
    d.elenco.forEach(function (x, i) {
      if (i) { L.aggiornaEsame(st, d, x.id, "esame", "contenuto_diverso"); }
    });
    verifica("«no» rifiutato se ne resta una non esaminata", L.impostaEsitoDecontam(st, d, "esito", "no").length > 0);
    L.aggiornaEsame(st, d, d.elenco[0].id, "esame", "contenuto_diverso");
    verifica("«no» ammesso con tutte esaminate come contenuto diverso e materiale completo",
             L.impostaEsitoDecontam(st, d, "esito", "no").length === 0
             && L.controlloDecontam(d, L.voceDecontam(st, d)).fase === "completo");
    var r = L.aggiornaEsame(st, d, d.elenco[1].id, "esame", "stesso_contenuto");
    verifica("cambiare un esame dopo il «no» annulla il «no»", r.avvisi.length > 0 && L.voceDecontam(st, d).esito === L.NON);
    L.impostaEsitoDecontam(st, d, "esito", "si");
    verifica("«sì» senza la corrispondenza decisiva resta incompleto",
             L.controlloDecontam(d, L.voceDecontam(st, d)).fase === "in_corso");
    L.impostaEsitoDecontam(st, d, "decisiva", d.elenco[1].id);
    var s = L.controlloDecontam(d, L.voceDecontam(st, d));
    verifica("«sì» con la corrispondenza decisiva esaminata come stesso contenuto: completo, le altre restano distinte",
             s.fase === "completo" && L.voceDecontam(st, d).corrispondenze[d.elenco[0].id].esame === "contenuto_diverso");
    L.impostaEsitoDecontam(st, d, "decisiva", d.elenco[0].id);
    verifica("una decisiva non esaminata come stesso contenuto è un errore",
             L.controlloDecontam(d, L.voceDecontam(st, d)).errori.length > 0);
    var riga = L.tsvDecontaminazione(P, st).split("\n").filter(function (x) { return x.indexOf(d.id + "\t") === 0; })[0];
    verifica("TSV: esito incompleto segnalato, corrispondenze guardate elencate",
             riga.split("\t")[2] === "incompleto_si" && riga.indexOf(d.elenco[0].id + "=contenuto_diverso") >= 0, riga);
    var finto = L.copia(d);
    finto.materiale = "incompleto: prova";
    verifica("materiale incompleto: il «no» non è ammesso", L.puoEscludere(finto, null).join().indexOf("materiale") >= 0);
    var somiglianza = L.voceDecontam(L.nuovoStato(P), d, true);
    verifica("la somiglianza dei pixel non compila nessun giudizio", somiglianza.esito === L.NON
             && Object.keys(somiglianza.corrispondenze).every(function (k) {
               return somiglianza.corrispondenze[k].esame === "non_esaminata"; }));
  });

  /* ------------------------------------------------------------ 7. interfaccia (DOM) */
  prova("interfaccia", function () {
    var st = L.nuovoStato(P);
    App.inizializza(P, S, E, st, {persisti: false});
    var u = P.campione.filter(function (x) {  // un target riconoscibile, per esempio delle coordinate "(2, -3)"
      var t = S[x.id].varianti.length ? S[x.id].varianti[0].risposta : "";
      return x.livello === "qa" && !x.coppia && t && (/^\(/.test(t) || t.length >= 6) && x.prompt.indexOf(t) < 0;
    })[0];
    var nodo = App.costruisciVistaUnita(u);
    var risposta = S[u.id].varianti[0].risposta;
    verifica("prima del giudizio: nessuna soluzione, opzione ausiliaria o materiale nella pagina",
             !nodo.querySelector(".soluzione") && !nodo.querySelector("table.soluzioni")
             && !nodo.querySelector("details.materiale") && nodo.textContent.indexOf(risposta) < 0,
             u.id + " · target " + risposta);
    verifica("il prompt è esatto, con gli a capo", nodo.querySelector("pre.prompt").textContent === u.prompt);
    L.aggiornaBozza(st, u, "", "risposta", "prova");
    L.aggiornaBozza(st, u, "", "leggibilita", "si");
    L.salvaPrima(st, u);
    L.mostraSoluzione(st, u);
    nodo = App.costruisciVistaUnita(u);
    verifica("dopo il giudizio: soluzione e materiale visibili, primo giudizio bloccato",
             !!nodo.querySelector(".soluzione") && nodo.textContent.indexOf(risposta) >= 0
             && Array.prototype.every.call(nodo.querySelectorAll(".prima textarea, .prima select"), function (x) {
               return x.disabled; }));
    var c = P.campione.filter(function (x) { return x.coppia; })[0];
    nodo = App.costruisciVistaUnita(c);
    verifica("coppia: due immagini e due moduli del primo giudizio (a e b)",
             nodo.querySelectorAll(".galleria img").length === 2 && nodo.querySelectorAll(".prima fieldset").length === 2);
    var d = P.decontaminazione.filter(function (x) { return x.elenco.length > 10; })[0];
    nodo = App.costruisciVistaDecontam(d);
    verifica("decontaminazione: record accanto a tutte le immagini candidate, con gli identificativi",
             nodo.querySelectorAll(".corrispondenza").length === d.elenco.length
             && d.elenco.every(function (x) { return nodo.querySelector("#c-" + x.id); }), d.id + " · " + d.elenco.length);
    verifica("riepilogo costruito", !!App.costruisciRiepilogo().querySelector("table.riepilogo"));
  });

  disegna();

  /* ------------------------------------------------------------ 8. immagini (asincrono) */
  var percorsi = {};
  P.campione.concat(P.mirati).forEach(function (u) { u.immagini.forEach(function (im) { percorsi[im.immagine] = 1; }); });
  P.decontaminazione.forEach(function (d) {
    percorsi[d.immagine] = 1;
    d.elenco.forEach(function (x) { if (x.immagine) { percorsi[x.immagine] = 1; } });
  });
  var elenco = Object.keys(percorsi), caricate = 0, errori = [];
  var riga = {nome: "immagini: tutte caricate dalla cartella", ok: false, dettaglio: "in corso…"};
  righe.push(riga);
  disegna();
  elenco.forEach(function (p) {
    var img = new Image();
    img.onload = function () { caricate += 1; fine(); };
    img.onerror = function () { errori.push(p); fine(); };
    img.src = p;
  });
  function fine() {
    if (caricate + errori.length < elenco.length) {
      return;
    }
    riga.ok = !errori.length && caricate === elenco.length;
    riga.dettaglio = caricate + " su " + elenco.length + (errori.length ? "; non trovate: " + errori.slice(0, 5).join(", ") : "");
    disegna();
  }
})(window);
