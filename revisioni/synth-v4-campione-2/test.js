/* Autotest di verifica.html: logica e interfaccia sui dati veri del pacchetto, senza toccare lo stato salvato nel
   browser (App.inizializza con persisti: false o con una memoria finta; scaricamenti e conferme finti). Senza DOM, cioè
   fuori da un browser, le parti che chiedono il DOM o le immagini risultano saltate, non superate; con il DOM simulato
   di scripts/review_package_dom.js (esecuzione con Duktape sul cluster) si salta solo il caricamento delle immagini.

   L'output non mostra mai soluzioni: nomi e dettagli contengono solo identificativi e conteggi, e alla fine un
   controllo cerca nell'output le risposte attese, le opzioni ausiliarie, i motivi e le misure dei casi (non presenti
   nel testo visibile dell'unità, di almeno 4 caratteri); se ne trova, nasconde il dettaglio e fallisce. */
(function (root) {
  "use strict";
  var L = root.Logica, P = root.PACCHETTO, S = root.SOLUZIONI, E = root.EVIDENZE, App = root.App;
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

  var SEGRETI = null;
  function segreti() {
    if (SEGRETI) {
      return SEGRETI;
    }
    var out = {};
    P.campione.concat(P.mirati).forEach(function (u) {
      var s = S[u.id];
      if (!s) {
        return;
      }
      var visibile = [u.id, u.prompt || "", u.domanda || "", u.consegna || ""].concat(u.immagini.map(function (im) {
        return im.record; })).join("\n");
      var cand = [s.motivo, s.misura];
      s.varianti.forEach(function (x) { cand = cand.concat([x.risposta], x.opzioni); });
      cand.forEach(function (t) {
        t = String(t === null || t === undefined ? "" : t).trim();
        if (t.length >= 4 && !confinato(visibile, t)) {
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

  /* Nasconde nome e dettaglio delle righe che contengono una soluzione e rimette in coda la riga del controllo. */
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
    rigaSegreti = {nome: "output dell'autotest senza risposte attese, opzioni ausiliarie o annotazioni risolutive",
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

  if (!L || !P || !S || !E || !App) {
    verifica("dati e script caricati (dati/*.js, logica.js, app.js)", false, "manca qualcosa: estrai tutta la cartella");
    disegna();
    return;
  }
  verifica("dati e script caricati", true, P.id + " v" + P.versione + " · impronta " + P.impronta.slice(0, 16) +
           (simulato ? " · DOM simulato" : ""));
  var C = L.contesto(P, E);

  /* ------------------------------------------------------------ strumenti */

  function unita(sezione, filtro) {
    return P[sezione].filter(filtro)[0];
  }

  function libera(sezione, filtro) {  // un'unità senza collegamenti
    return unita(sezione, function (x) { return !C.collegati[x.id].length && filtro(x); });
  }

  function riempiPrima(st, u, risposta) {
    u.varianti.forEach(function (k) {
      L.aggiornaBozza(st, u, k, "risposta", risposta + (k ? "-" + k : ""));
      L.aggiornaBozza(st, u, k, "leggibilita", "si");
    });
    return L.salvaPrima(st, u);
  }

  function riempiCollegate(st, u, risposta) {
    C.collegati[u.id].forEach(function (c) {
      var w = st.unita[c.id];
      if (!w || !w.prima) {
        riempiPrima(st, C.idx.unita[c.id], risposta);
      }
    });
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

  function esposto(st, record, tramite) {
    return st.esposizioni.some(function (e) { return e.record === record && e.tramite === tramite; });
  }

  function rigaTsv(st, id, variante) {
    return (L.tsvEsiti(C, st).split("\n").filter(function (r) {
      return r.indexOf(id + "\t" + variante + "\t") === 0; })[0] || "").split("\t");
  }

  /* due unità che mostrano lo stesso record: l'esempio della revisione (C-test-008 e C-test-012 su synth-test-00038)
     se c'è, altrimenti il primo record condiviso */
  var condivisi = Object.keys(C.unitaDi).filter(function (r) { return C.unitaDi[r].length > 1; });
  var recordComune = condivisi.indexOf("synth-test-00038") >= 0 ? "synth-test-00038" : condivisi[0];
  var U1 = recordComune ? C.idx.unita[C.unitaDi[recordComune][0]] : null;
  var U2 = recordComune ? C.idx.unita[C.unitaDi[recordComune][1]] : null;

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

  /* ------------------------------------------------------------ 2. collegamenti fra unità e record */
  prova("collegamenti", function () {
    var tutte = P.campione.concat(P.mirati);
    var scene = Object.keys(C.recordDiScena).filter(function (k) { return C.recordDiScena[k].length > 1; });
    var conColl = tutte.filter(function (u) { return C.collegati[u.id].length; });
    verifica("collegamenti: record mostrati da più unità e scene uguali fra record diversi", condivisi.length > 0,
             condivisi.length + " record condivisi, " + scene.length + " scene con più record, " + conColl.length +
             " unità collegate");
    verifica("ogni record del pacchetto ha lo stato della scena (dal JSON del record)",
             Object.keys(C.unitaDi).every(function (r) { return C.scenaDi[r] !== undefined; }));
    var reciproci = tutte.every(function (u) {
      return C.collegati[u.id].every(function (c) {
        return C.collegati[c.id].some(function (x) { return x.id === u.id; }); });
    });
    verifica("collegamenti reciproci", reciproci);
    var perRecord = condivisi.every(function (r) {
      return C.unitaDi[r].every(function (a) {
        return C.unitaDi[r].every(function (b) {
          return a === b || C.collegati[a].some(function (x) { return x.id === b; }); }); });
    });
    verifica("unità sullo stesso record collegate fra loro", perRecord, U1 && U2 ? U1.id + " e " + U2.id + " (" +
             recordComune + ")" : "");
    var perScena = scene.every(function (k) {
      var us = [];
      C.recordDiScena[k].forEach(function (r) { us = us.concat(C.unitaDi[r]); });
      return us.every(function (a) {
        return us.every(function (b) { return a === b || C.collegati[a].some(function (x) { return x.id === b; }); });
      });
    });
    verifica("unità su record con la stessa scena collegate fra loro", perScena, scene.length + " scene");
    var coppia = tutte.filter(function (u) { return u.coppia; })[0];
    verifica("coppie: la soluzione espone i record di entrambe le varianti", C.recordDi[coppia.id].length === 2
             && C.recordDi[coppia.id].join() === [coppia.record_per_variante.a, coppia.record_per_variante.b].join(),
             coppia.id);
  });

  /* ------------------------------------------------------------ 3. primo giudizio separato dalla soluzione */
  prova("primo giudizio", function () {
    var u = libera("campione", function (x) { return x.livello === "qa" && !x.coppia; });
    var st = L.nuovoStato(P);
    verifica("la soluzione non si mostra senza un primo giudizio salvato", L.mostraSoluzione(C, st, u).length > 0);
    verifica("il primo giudizio vuoto non si salva", L.salvaPrima(st, u).length > 0 && L.voce(st, u).prima === null);
    L.aggiornaBozza(st, u, "", "risposta", "right");
    L.aggiornaBozza(st, u, "", "leggibilita", "si");
    verifica("il primo giudizio completo si salva", L.salvaPrima(st, u).length === 0 && L.voce(st, u).prima !== null);
    L.aggiornaBozza(st, u, "", "risposta", "left");
    verifica("con modifiche non salvate la soluzione resta nascosta", !L.puoMostrare(C, st, u).ok);
    L.aggiornaBozza(st, u, "", "risposta", "right");
    verifica("dopo il salvataggio la soluzione si mostra", L.mostraSoluzione(C, st, u).length === 0
             && !!L.voce(st, u).soluzione_mostrata_il && esposto(st, u.immagini[0].record, u.id));
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
    verifica("nel TSV: non_riesco_a_rispondere", rigaTsv(st, v.id, v.varianti[0])[3] === "non_riesco_a_rispondere");
  });

  /* ------------------------------------------------------------ 4. stesso record: la prima rivelazione */
  prova("stesso record", function () {
    if (!U1) {
      verifica("due unità sullo stesso record", false, "nessun record condiviso nel pacchetto");
      return;
    }
    var st = L.nuovoStato(P);
    var k2 = U2.varianti[0];
    riempiPrima(st, U1, "uno");
    var p = L.puoMostrare(C, st, U1);
    verifica("stesso record: senza il primo giudizio dell'altra unità la soluzione non si mostra",
             !p.ok && p.mancanti.indexOf(U2.id) >= 0 && L.mostraSoluzione(C, st, U1).length > 0
             && !L.voce(st, U1).soluzione_mostrata_il, U1.id + " → " + p.mancanti.join(", "));
    L.aggiornaBozza(st, U2, k2, "risposta", "bozza non salvata");
    verifica("stesso record: una bozza non salvata dell'altra unità non basta",
             L.puoMostrare(C, st, U1).mancanti.indexOf(U2.id) >= 0);
    riempiPrima(st, U2, "due");
    L.aggiornaBozza(st, U2, k2, "note", "modifica non salvata");
    verifica("stesso record: modifiche non salvate dell'altra unità bloccano la soluzione",
             L.puoMostrare(C, st, U1).non_salvate.indexOf(U2.id) >= 0);
    L.aggiornaBozza(st, U2, k2, "note", "");
    riempiCollegate(st, U1, "collegata");
    verifica("stesso record: con i primi giudizi delle unità collegate salvati la soluzione si mostra",
             L.puoMostrare(C, st, U1).ok && L.mostraSoluzione(C, st, U1).length === 0);
    var v2 = L.voce(st, U2);
    verifica("stesso record: alla rivelazione il primo giudizio dell'altra unità resta fissato, con provenienza",
             !!v2.prima_fissata && v2.prima_fissata.motivo === "record_esposto" && v2.prima_fissata.tramite === U1.id
             && v2.prima_fissata.record.indexOf(recordComune) >= 0 && !v2.soluzione_mostrata_il, U2.id);
    verifica("stesso record: l'esposizione è registrata (record, quando, tramite)",
             esposto(st, recordComune, U1.id) && st.esposizioni.every(function (e) {
               return typeof e.quando === "string" && e.origine === "soluzione_mostrata"; }));
    verifica("stesso record: il primo giudizio fissato non si modifica e non si risalva",
             L.aggiornaBozza(st, U2, k2, "risposta", "cambio").length > 0 && L.salvaPrima(st, U2).length > 0
             && v2.prima.varianti[k2].risposta === "due" + (k2 ? "-" + k2 : ""));
    var s2 = L.statoUnita(U2, v2);
    verifica("stesso record: l'altra unità resta indipendente (salvata prima) e con la soluzione da mostrare",
             s2.fissata && !s2.non_indipendente && s2.fase === "primo_giudizio");
    verifica("stesso record: la sua soluzione si mostra poi normalmente", L.mostraSoluzione(C, st, U2).length === 0);
    verifica("stesso record: nel TSV la risposta resta quella indipendente",
             rigaTsv(st, U2.id, k2)[3] === "due" + (k2 ? "-" + k2 : ""));
  });

  /* ------------------------------------------------------------ 5. coppie */
  prova("coppie", function () {
    var u = libera("campione", function (x) { return x.coppia; });
    var st = L.nuovoStato(P);
    L.aggiornaBozza(st, u, "a", "risposta", "9");
    L.aggiornaBozza(st, u, "a", "leggibilita", "si");
    verifica("coppia: il primo giudizio chiede entrambe le varianti", L.salvaPrima(st, u).length > 0);
    L.aggiornaBozza(st, u, "b", "risposta", "11");
    L.aggiornaBozza(st, u, "b", "leggibilita", "incerto");
    verifica("coppia: con a e b il primo giudizio si salva", L.salvaPrima(st, u).length === 0);
    L.mostraSoluzione(C, st, u);
    verifica("coppia: la soluzione espone i record di a e di b",
             esposto(st, u.record_per_variante.a, u.id) && esposto(st, u.record_per_variante.b, u.id));
    riempiDopo(st, u);
    L.aggiornaDopo(st, u, "b", "corretta", "no");
    L.aggiornaDopo(st, u, "coppia", "contratto_coppia", "no");
    var v = L.voce(st, u);
    verifica("coppia: risposte e giudizi distinti per a e b", v.prima.varianti.a.risposta === "9"
             && v.prima.varianti.b.risposta === "11" && v.dopo.varianti.a.corretta === "si"
             && v.dopo.varianti.b.corretta === "no");
    var ra = rigaTsv(st, u.id, "a"), rb = rigaTsv(st, u.id, "b"), rc = rigaTsv(st, u.id, "coppia");
    verifica("coppia: righe a, b e coppia nel TSV, con risposte e contratto",
             ra[3] === "9" && rb[3] === "11" && rb[4] === "no" && rc[8] === "no" && ra[8] === "non_applicabile");
    verifica("coppia: completa con i giudizi di a, b e del contratto", L.statoUnita(u, v).fase === "completo");
    var pc = P.campione.concat(P.mirati).filter(function (x) { return x.coppia && C.collegati[x.id].length; })[0];
    if (!pc) {
      salta("coppia collegata ad altre unità", "nessuna coppia con collegamenti");
      return;
    }
    var st2 = L.nuovoStato(P);
    riempiPrima(st2, pc, "p");
    var b = L.puoMostrare(C, st2, pc);
    verifica("coppia collegata: la soluzione aspetta i primi giudizi delle unità collegate a uno dei suoi record",
             !b.ok && b.mancanti.length === C.collegati[pc.id].length, pc.id + " → " + b.mancanti.join(", "));
    riempiCollegate(st2, pc, "q");
    L.mostraSoluzione(C, st2, pc);
    verifica("coppia collegata: alla rivelazione le unità collegate restano fissate tramite la coppia",
             C.collegati[pc.id].every(function (c) {
               var w = st2.unita[c.id];
               return w.prima_fissata && w.prima_fissata.tramite === pc.id; }), C.collegati[pc.id].map(function (c) {
               return c.id; }).join(", "));
    var mm = P.mirati.filter(function (x) { return C.recordDi[x.id].length > 1 && !x.coppia; })[0];
    if (mm) {
      var st3 = L.nuovoStato(P);
      riempiPrima(st3, mm, "m");
      riempiCollegate(st3, mm, "n");
      L.mostraSoluzione(C, st3, mm);
      verifica("caso mirato con più record: la soluzione li espone tutti e fissa le unità collegate a ciascuno",
               C.recordDi[mm.id].every(function (r) { return esposto(st3, r, mm.id); }) && C.collegati[mm.id].every(
                 function (c) { return !!st3.unita[c.id].prima_fissata; }), mm.id + " · " + C.recordDi[mm.id].length +
               " record");
    }
  });

  /* ------------------------------------------------------------ 6. casi sulla scena intera */
  prova("scene", function () {
    var u = unita("mirati", function (x) { return x.livello === "scena"; });
    var st = L.nuovoStato(P);
    L.aggiornaBozza(st, u, "", "leggibilita", "si");
    verifica("caso sulla scena intera: basta il giudizio iniziale", L.salvaPrima(st, u).length === 0, u.id);
  });

  /* ------------------------------------------------------------ 7. esportazione e importazione */
  var speciale = "√2 · x² ≤ 3 «virgolette» \"doppie\" 'singole'\tTAB\nseconda riga\r\nterza\\barra 😀 è à ∠ABC = 90°";
  prova("esportazione", function () {
    var st = L.nuovoStato(P);
    var u = libera("campione", function (x) { return x.livello === "qa" && !x.coppia; });
    L.aggiornaBozza(st, u, "", "risposta", speciale);
    L.aggiornaBozza(st, u, "", "note", speciale);
    L.aggiornaBozza(st, u, "", "leggibilita", "si");
    L.salvaPrima(st, u);
    L.mostraSoluzione(C, st, u);
    L.aggiornaDopo(st, u, "", "note", speciale);
    L.aggiornaDopo(st, u, "", "risposta_dopo_soluzione", speciale);
    L.aggiornaDopo(st, u, "", "corretta", "non_verificabile");
    var c = libera("campione", function (x) { return x.coppia; });
    riempiPrima(st, c, "x");
    if (U1) {
      riempiPrima(st, U1, "e1");
      riempiCollegate(st, U1, "e2");
      L.mostraSoluzione(C, st, U1);
    }
    var d = P.decontaminazione[0];
    L.aggiornaEsame(st, d, d.elenco[0].id, "note", speciale);
    L.impostaEsitoDecontam(st, d, "note", speciale);
    st.revisore = "Revisore è";
    var testo = L.esportaJSON(C, st);
    var r = L.controllaImport(C, testo);
    verifica("il file esportato si reimporta senza errori", !r.errori.length, r.errori.join("; "));
    verifica("reimportato identico: risposte, note su più righe, caratteri speciali, giudizi, coppie, esposizioni, " +
             "primi giudizi fissati, decontaminazione",
             r.stato && L.canonico(r.stato.unita) === L.canonico(st.unita)
             && L.canonico(r.stato.decontaminazione) === L.canonico(st.decontaminazione)
             && L.canonico(r.stato.esposizioni) === L.canonico(st.esposizioni)
             && L.canonico(r.stato.archivio) === L.canonico(st.archivio)
             && r.stato.unita[u.id].prima.varianti[""].risposta === speciale
             && r.stato.unita[u.id].dopo.varianti[""].risposta_dopo_soluzione === speciale
             && r.stato.revisore === "Revisore è" && !r.migrazione);
    verifica("celle TSV: niente tabulazioni o a capo, e si rileggono uguali",
             !/[\t\n\r]/.test(L.cella(speciale)) && L.leggiCella(L.cella(speciale)) === speciale.replace(/\r\n/g, "\n"));
    var tsv = L.tsvEsiti(C, st).split("\n");
    verifica("esiti.tsv: intestazione dei moduli e 655 righe da 13 colonne",
             tsv[0] === L.TSV_ESITI.join("\t") && tsv.length === 657 && tsv[656] === ""
             && tsv.slice(1, 656).every(function (x) { return x.split("\t").length === 13; }));
    var riga = rigaTsv(st, u.id, "");
    verifica("esiti.tsv: risposta e note con i caratteri speciali conservati",
             L.leggiCella(riga[3]) === speciale.replace(/\r\n/g, "\n") && L.leggiCella(riga[12]) === speciale.replace(/\r\n/g, "\n"));
    var non = rigaTsv(st, P.campione[5].id, P.campione[5].varianti[0]);
    verifica("esiti.tsv: un'unità non toccata è non_compilato, non approvata", non[3] === "non_compilato"
             && non[4] === "non_compilato" && non[9] === "non_compilato");
    var dt = L.tsvDecontaminazione(C, st).split("\n");
    verifica("esiti_decontaminazione.tsv: intestazione e 264 righe da 6 colonne",
             dt[0] === L.TSV_DECONTAMINAZIONE.join("\t") && dt.length === 266 && dt.slice(1, 265).every(function (x) {
               return x.split("\t").length === 6; }));
  });

  prova("importazione rifiutata", function () {
    var st = L.nuovoStato(P);
    var u = P.campione[0];
    riempiPrima(st, u, "a");
    var buono = JSON.parse(L.esportaJSON(C, st));
    function rifiuta(nome, modifica) {
      var x = JSON.parse(JSON.stringify(buono));
      modifica(x);
      verifica("importazione rifiutata: " + nome, L.controllaImport(C, JSON.stringify(x)).errori.length > 0);
    }
    rifiuta("impronta di un altro pacchetto", function (x) { x.pacchetto.impronta = new Array(65).join("0"); });
    rifiuta("versione diversa", function (x) { x.pacchetto.versione = "0"; });
    rifiuta("formato sconosciuto", function (x) { x.schema = "altro/1"; });
    rifiuta("unità sconosciuta", function (x) { x.unita["C-coord-999"] = x.unita[u.id]; });
    rifiuta("record diverso", function (x) { x.unita[u.id].varianti[u.varianti[0]].record = "synth-0000000"; });
    rifiuta("valore non ammesso", function (x) { x.unita[u.id].dopo.varianti[u.varianti[0]].corretta = "forse"; });
    rifiuta("soluzione mostrata senza primo giudizio", function (x) {
      x.unita[u.id].prima = null;
      x.unita[u.id].soluzione_mostrata_il = "2026-11-05T10:00:00.000Z";
    });
    rifiuta("esposizione con record e unità che non si corrispondono", function (x) {
      x.esposizioni.push({record: "synth-test-99999", quando: "2026-11-05T10:00:00.000Z", tramite: u.id,
                          origine: "soluzione_mostrata"});
    });
    rifiuta("formato 2 senza il registro delle esposizioni", function (x) { delete x.esposizioni; });
    rifiuta("primo giudizio fissato non valido", function (x) {
      x.unita[u.id].prima_fissata = {quando: "2026-11-05T10:00:00.000Z", motivo: "altro", tramite: u.id,
                                     record: [u.immagini[0].record], collegamento: "stesso_record"};
    });
    verifica("un file non JSON è rifiutato", L.controllaImport(C, "non è json").errori.length > 0);
  });

  /* stato nel formato 1 (pacchetto a9252b5): le voci senza i campi nuovi */
  function formato1(st) {
    var x = JSON.parse(L.esportaJSON(C, st));
    x.schema = L.SCHEMA_V1;
    delete x.esposizioni;
    delete x.archivio;
    delete x.migrazioni;
    Object.keys(x.unita).forEach(function (id) {
      delete x.unita[id].prima_fissata;
      delete x.unita[id].esposta_senza_primo_giudizio;
    });
    return x;
  }

  prova("formato 1", function () {
    if (!U1) {
      return;
    }
    var k2 = U2.varianti[0];
    function caso(nomeCaso, conU2, salvataU2) {
      var st = L.nuovoStato(P);
      riempiPrima(st, U1, "uno");
      var v1 = st.unita[U1.id];
      v1.prima.salvata_il = "2026-11-05T09:00:00.000Z";
      v1.soluzione_mostrata_il = "2026-11-05T10:00:00.000Z";  // nel formato 1 la soluzione si mostrava da sola
      if (conU2) {
        riempiPrima(st, U2, "due");
        st.unita[U2.id].prima.salvata_il = salvataU2;
      }
      var r = L.controllaImport(C, JSON.stringify(formato1(st)));
      return r;
    }
    var r0 = caso("senza", false, null);
    var w0 = r0.stato && r0.stato.unita[U2.id];
    verifica("formato 1: si importa con una migrazione esplicita, la soluzione mostrata diventa un'esposizione",
             !r0.errori.length && !!r0.migrazione && r0.stato.schema === L.SCHEMA_STATO
             && esposto(r0.stato, recordComune, U1.id) && r0.stato.migrazioni.length === 1, r0.errori.join("; "));
    verifica("formato 1: esposizione senza primo giudizio registrata, il giudizio successivo non è indipendente",
             !!w0 && !!w0.esposta_senza_primo_giudizio && w0.esposta_senza_primo_giudizio.tramite === U1.id
             && !w0.prima && L.statoUnita(U2, w0).non_indipendente, U2.id);
    L.aggiornaBozza(r0.stato, U2, k2, "risposta", "tardiva");
    L.aggiornaBozza(r0.stato, U2, k2, "leggibilita", "si");
    U2.varianti.slice(1).forEach(function (k) {
      L.aggiornaBozza(r0.stato, U2, k, "risposta", "tardiva");
      L.aggiornaBozza(r0.stato, U2, k, "leggibilita", "si");
    });
    var salvato = L.salvaPrima(r0.stato, U2).length === 0;
    var riga = rigaTsv(r0.stato, U2.id, k2);
    verifica("formato 1: il primo giudizio salvato dopo l'esposizione resta fissato e non indipendente",
             salvato && r0.stato.unita[U2.id].prima_fissata.motivo === "record_gia_esposto"
             && L.aggiornaBozza(r0.stato, U2, k2, "risposta", "x").length > 0);
    verifica("formato 1: nel TSV la risposta non indipendente non va nella colonna della risposta",
             riga[3] === "non_indipendente" && L.leggiCella(riga[12]).indexOf("non indipendente") >= 0
             && L.leggiCella(riga[12]).indexOf("risposta: tardiva") >= 0);
    var rDopo = caso("dopo", true, "2026-11-05T11:00:00.000Z");
    var wd = rDopo.stato.unita[U2.id];
    verifica("formato 1: primo giudizio salvato dopo la soluzione dell'altra unità: fissato e non indipendente",
             !!wd.prima_fissata && wd.prima_fissata.motivo === "record_gia_esposto" && !!wd.esposta_senza_primo_giudizio);
    var rPrima = caso("prima", true, "2026-11-05T09:30:00.000Z");
    var wp = rPrima.stato.unita[U2.id];
    verifica("formato 1: primo giudizio salvato prima: fissato e indipendente, risposta conservata",
             !!wp.prima_fissata && wp.prima_fissata.motivo === "record_esposto" && !wp.esposta_senza_primo_giudizio
             && wp.prima.varianti[k2].risposta === "due" + (k2 ? "-" + k2 : ""));
    var semplice = L.nuovoStato(P);
    var lu = libera("campione", function (x) { return !x.coppia; });
    riempiPrima(semplice, lu, "solo");
    var rs = L.controllaImport(C, JSON.stringify(formato1(semplice)));
    verifica("formato 1 senza soluzioni mostrate: nessuna esposizione e lavoro conservato",
             !rs.errori.length && !rs.stato.esposizioni.length
             && L.canonico(rs.stato.unita[lu.id].prima) === L.canonico(semplice.unita[lu.id].prima));
  });

  /* ------------------------------------------------------------ 8. unione: file vecchi, file nuovi, conflitti */
  prova("unione", function () {
    if (!U1) {
      return;
    }
    var st = L.nuovoStato(P);
    riempiPrima(st, U1, "uno");
    riempiCollegate(st, U1, "due");
    var vecchio = L.controllaImport(C, L.esportaJSON(C, st)).stato;  // esportato prima della rivelazione
    L.mostraSoluzione(C, st, U1);
    L.aggiornaDopo(st, U1, U1.varianti[0], "note", "nota del giudizio finale");
    ["senza_scelta", "browser", "file", "sostituisci"].forEach(function (modo) {
      var m = L.unisci(C, st, vecchio, modo);
      var w = m.stato.unita[U2.id], v = m.stato.unita[U1.id];
      verifica("file vecchio (" + modo + "): esposizione, soluzione mostrata e primi giudizi fissati restano",
               esposto(m.stato, recordComune, U1.id) && v.soluzione_mostrata_il === st.unita[U1.id].soluzione_mostrata_il
               && !!v.prima_fissata && !!w.prima_fissata && !m.conflitti.length
               && L.aggiornaBozza(m.stato, U1, U1.varianti[0], "risposta", "x").length > 0
               && L.aggiornaBozza(m.stato, U2, U2.varianti[0], "risposta", "x").length > 0
               && v.dopo.varianti[U1.varianti[0]].note === "nota del giudizio finale", m.conflitti.length + " conflitti");
    });
    var nuovo = L.controllaImport(C, L.esportaJSON(C, st)).stato;
    var m2 = L.unisci(C, vecchio, nuovo, "browser");
    verifica("file più recente con un'esposizione in uno stato vecchio: esposizione e blocchi si importano anche " +
             "tenendo il browser", esposto(m2.stato, recordComune, U1.id) && !!m2.stato.unita[U1.id].soluzione_mostrata_il
             && !!m2.stato.unita[U2.id].prima_fissata && m2.esposizioni_nuove > 0 && !m2.conflitti.length);
    var divergente = L.copia(vecchio);
    U2.varianti.forEach(function (k) { divergente.unita[U2.id].prima.varianti[k].risposta = "rifatta altrove"; });
    divergente.unita[U2.id].bozza = L.copia(divergente.unita[U2.id].prima.varianti);
    divergente.unita[U2.id].prima.salvata_il = "2099-01-01T00:00:00.000Z";
    var m3 = L.unisci(C, divergente, nuovo, "senza_scelta");
    var a3 = m3.stato.archivio.filter(function (a) { return a.id === U2.id && a.origine === "browser"; });
    verifica("primo giudizio diverso salvato dopo l'esposizione: vale quello fissato, l'altro va nell'archivio come " +
             "non indipendente", m3.stato.unita[U2.id].prima.varianti[U2.varianti[0]].risposta.indexOf("due") === 0
             && a3.length === 1 && a3[0].motivo.indexOf("non indipendente") >= 0
             && m3.risolti.some(function (x) { return x.id === U2.id; }));
    var u = libera("campione", function (x) { return !x.coppia; });
    var a = L.nuovoStato(P), b = L.nuovoStato(P);
    riempiPrima(a, u, "uno");
    riempiPrima(b, u, "tre");
    var c = L.unisci(C, a, b, "senza_scelta");
    verifica("conflitto: risposte diverse per la stessa unità non si uniscono senza una scelta",
             c.conflitti.length === 1 && c.conflitti[0].id === u.id
             && c.stato.unita[u.id].prima.varianti[u.varianti[0]].risposta === "uno" && !c.archiviate);
    var cb = L.unisci(C, a, b, "browser"), cf = L.unisci(C, a, b, "file");
    verifica("conflitto: la versione scartata resta nell'archivio, sia tenendo il browser sia usando il file",
             cb.stato.unita[u.id].prima.varianti[u.varianti[0]].risposta === "uno"
             && cb.stato.archivio.some(function (x) { return x.id === u.id && x.origine === "file"; })
             && cf.stato.unita[u.id].prima.varianti[u.varianti[0]].risposta === "tre"
             && cf.stato.archivio.some(function (x) { return x.id === u.id && x.origine === "browser"; }));
    var a2 = L.copia(a);
    a2.unita[u.id].tempo_ms += 5000;
    var c2 = L.unisci(C, a, a2, "senza_scelta");
    verifica("solo il tempo diverso: nessun conflitto, si tiene il tempo maggiore", !c2.conflitti.length
             && c2.stato.unita[u.id].tempo_ms === a2.unita[u.id].tempo_ms);
    var c3 = L.unisci(C, L.nuovoStato(P), b, "senza_scelta");
    verifica("stato vuoto: il file si importa per intero", !c3.conflitti.length && c3.nuove === 1);
    var solo = L.nuovoStato(P);
    var w = libera("mirati", function () { return true; });
    riempiPrima(solo, w, "solo nel browser");
    var c4 = L.unisci(C, solo, b, "sostituisci");
    verifica("sostituzione: le voci solo nel browser e non fissate vanno nell'archivio, non si perdono",
             !c4.stato.unita[w.id] && c4.stato.archivio.some(function (x) { return x.id === w.id; }) && c4.tolte === 1);
  });

  /* ------------------------------------------------------------ 9. decontaminazione */
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
    var riga = (L.tsvDecontaminazione(C, st).split("\n").filter(function (x) { return x.indexOf(d.id + "\t") === 0; })[0]
                || "").split("\t");
    verifica("TSV: esito incompleto segnalato, corrispondenze guardate elencate",
             riga[2] === "incompleto_si" && riga[3].indexOf(d.elenco[0].id + "=contenuto_diverso") >= 0, d.id);
    var finto = L.copia(d);
    finto.materiale = "incompleto: prova";
    verifica("materiale incompleto: il «no» non è ammesso", L.puoEscludere(finto, null).join().indexOf("materiale") >= 0);
    var somiglianza = L.voceDecontam(L.nuovoStato(P), d, true);
    verifica("la somiglianza dei pixel non compila nessun giudizio", somiglianza.esito === L.NON
             && Object.keys(somiglianza.corrispondenze).every(function (k) {
               return somiglianza.corrispondenze[k].esame === "non_esaminata"; }));
  });

  /* ------------------------------------------------------------ 10. interfaccia (DOM) */
  var scaricati = [];
  function opzioniProva(memoria) {
    return {persisti: !!memoria, memoria: memoria || null, conferma: function () { return true; },
            scarica: function (nome, testo) { scaricati.push({nome: nome, byte: testo.length}); }};
  }
  function memoriaFinta(iniziale) {
    var dati = {};
    Object.keys(iniziale || {}).forEach(function (k) { dati[k] = iniziale[k]; });
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
    App.inizializza(P, S, E, st, opzioniProva(null));
    var u = P.campione.filter(function (x) {  // un target riconoscibile, per esempio delle coordinate "(2, -3)"
      var t = S[x.id].varianti.length ? S[x.id].varianti[0].risposta : "";
      return x.livello === "qa" && !x.coppia && !C.collegati[x.id].length && t && (/^\(/.test(t) || t.length >= 6)
        && x.prompt.indexOf(t) < 0;
    })[0];
    var risposta = S[u.id].varianti[0].risposta;
    var nodo = App.costruisciVistaUnita(u);
    verifica("prima del giudizio: nessuna soluzione, opzione ausiliaria o materiale nella pagina",
             !nodo.querySelector(".soluzione") && !nodo.querySelector("table.soluzioni")
             && !nodo.querySelector("details.materiale") && nodo.textContent.indexOf(risposta) < 0, u.id);
    verifica("il controllo dell'output riconosce un target reale (prova interna, il target non si stampa)",
             trovaSegreti(u.id + " · target " + risposta) > 0 && trovaSegreti(u.id) === 0);
    verifica("il prompt è esatto, con gli a capo", nodo.querySelector("pre.prompt").textContent === u.prompt);
    L.aggiornaBozza(st, u, "", "risposta", "prova");
    L.aggiornaBozza(st, u, "", "leggibilita", "si");
    L.salvaPrima(st, u);
    nodo = App.costruisciVistaUnita(u);
    var bottone = nodo.querySelector("button.rivela");
    verifica("primo giudizio salvato: il pulsante della soluzione è attivo", !!bottone && !bottone.disabled);
    bottone.click();
    nodo = App.costruisciVistaUnita(u);
    verifica("dopo il clic: soluzione e materiale visibili, primo giudizio bloccato",
             !!App.stato().unita[u.id].soluzione_mostrata_il && !!nodo.querySelector(".soluzione")
             && nodo.textContent.indexOf(risposta) >= 0
             && Array.prototype.every.call(nodo.querySelectorAll(".prima textarea, .prima select"), function (x) {
               return x.disabled; }), u.id);
    var c = P.campione.filter(function (x) { return x.coppia; })[0];
    nodo = App.costruisciVistaUnita(c);
    verifica("coppia: due immagini e due moduli del primo giudizio (a e b)",
             nodo.querySelectorAll(".galleria img").length === 2 && nodo.querySelectorAll(".prima fieldset").length === 2);
    var d = P.decontaminazione.filter(function (x) { return x.elenco.length > 10; })[0];
    nodo = App.costruisciVistaDecontam(d);
    verifica("decontaminazione: record accanto a tutte le immagini candidate, con gli identificativi",
             nodo.querySelectorAll(".corrispondenza").length === d.elenco.length
             && d.elenco.every(function (x) { return nodo.querySelector("#c-" + x.id); }), d.id + " · " + d.elenco.length);
    verifica("riepilogo costruito, con il registro dei record esposti",
             !!App.costruisciRiepilogo().querySelector("table.riepilogo")
             && !!App.costruisciRiepilogo().querySelector("table.esposizioni"));
  });

  prova("interfaccia: unità collegate", function () {
    if (!U1) {
      return;
    }
    var st = L.nuovoStato(P);
    App.inizializza(P, S, E, st, opzioniProva(null));
    riempiPrima(st, U1, "uno");
    var nodo = App.costruisciVistaUnita(U1);
    var box = nodo.querySelector("div.collegate");
    var bottoni = box ? box.querySelectorAll("button.collegamento") : [];
    var ids = Array.prototype.map.call(bottoni, function (b) { return b.getAttribute("data-id"); });
    verifica("unità collegate: elencate con un pulsante per raggiungerle e la soluzione bloccata",
             ids.indexOf(U2.id) >= 0 && ids.length === C.collegati[U1.id].length
             && nodo.querySelector("button.rivela").disabled, U1.id + " → " + ids.join(", "));
    var voci = Array.prototype.map.call(box.querySelectorAll("li"), function (li) { return li.textContent; });
    var recordColl = [];
    C.collegati[U1.id].forEach(function (c) { recordColl = recordColl.concat(c.record_collegati, c.record_esposti); });
    verifica("unità collegate: nel primo passaggio non si dice, unità per unità, se il legame è lo stesso record o la " +
             "stessa scena, né quale record", voci.length === C.collegati[U1.id].length && voci.every(function (t) {
               return t.indexOf("stessa scena") < 0 && t.indexOf("stesso record") < 0 && recordColl.every(function (r) {
                 return t.indexOf(r) < 0; }); }));
    riempiCollegate(st, U1, "due");
    nodo = App.costruisciVistaUnita(U1);
    nodo.querySelector("button.rivela").click();
    var w = App.stato().unita[U2.id];
    verifica("unità collegate: il clic mostra la soluzione e fissa i loro primi giudizi",
             !!App.stato().unita[U1.id].soluzione_mostrata_il && !!w.prima_fissata && w.prima_fissata.tramite === U1.id);
    var dopo = App.costruisciVistaUnita(U1).querySelector("div.collegate");
    verifica("unità collegate: dopo la soluzione si dice il legame (stesso record o stessa scena) con i record",
             !!dopo && dopo.textContent.indexOf(U2.id + " · primo giudizio fissato · stesso record: " + recordComune) >= 0);
    nodo = App.costruisciVistaUnita(U2);
    verifica("unità collegata: campi del primo giudizio bloccati, provenienza mostrata, nessun pulsante di salvataggio",
             Array.prototype.every.call(nodo.querySelectorAll(".prima textarea, .prima select, .prima input"),
                                        function (x) { return x.disabled; })
             && !!nodo.querySelector(".prima p.fissato") && !nodo.querySelector("button.salva-prima")
             && !nodo.querySelector(".soluzione"), U2.id);
  });

  prova("interfaccia: importazione e sostituzione", function () {
    if (!U1) {
      return;
    }
    var st = L.nuovoStato(P);
    var u3 = libera("campione", function (x) { return !x.coppia && x.livello === "qa"; });
    riempiPrima(st, U1, "uno");
    riempiCollegate(st, U1, "due");
    riempiPrima(st, u3, "browser");
    var vecchio = JSON.parse(L.esportaJSON(C, st));
    vecchio.unita[u3.id].prima.varianti[u3.varianti[0]].risposta = "file vecchio";
    vecchio.unita[u3.id].bozza = L.copia(vecchio.unita[u3.id].prima.varianti);
    L.mostraSoluzione(C, st, U1);
    scaricati = [];
    App.inizializza(P, S, E, st, opzioniProva(null));
    var msg = document.createElement("p"), box = document.createElement("div");
    var u = App.importaTesto(JSON.stringify(vecchio), msg, box);
    var dopo = App.stato();
    verifica("importare un file vecchio: copia del browser scaricata prima, esposizione e blocchi restano",
             !!u && scaricati.length === 1 && scaricati[0].nome.indexOf("_copia_del_browser_") > 0
             && esposto(dopo, recordComune, U1.id) && !!dopo.unita[U1.id].soluzione_mostrata_il
             && !!dopo.unita[U2.id].prima_fissata);
    verifica("importare un file vecchio: il lavoro in conflitto non si sovrascrive senza una scelta",
             u.conflitti.length === 1 && u.conflitti[0].id === u3.id
             && dopo.unita[u3.id].prima.varianti[u3.varianti[0]].risposta === "browser"
             && !!box.querySelector("button.sostituisci"));
    box.querySelector("button.sostituisci").click();
    var fine = App.stato();
    verifica("«Sostituisci tutto con il file»: vale il file nei conflitti, ma esposizione e primi giudizi fissati " +
             "restano e la versione del browser va nell'archivio",
             fine.unita[u3.id].prima.varianti[u3.varianti[0]].risposta === "file vecchio"
             && esposto(fine, recordComune, U1.id) && !!fine.unita[U1.id].soluzione_mostrata_il
             && !!fine.unita[U2.id].prima_fissata && L.aggiornaBozza(fine, U1, U1.varianti[0], "risposta", "x").length > 0
             && fine.archivio.some(function (a) { return a.id === u3.id && a.origine === "browser"; }));
  });

  prova("interfaccia: ripresa nel browser", function () {
    if (!U1) {
      return;
    }
    var vecchio = L.nuovoStato(P);
    riempiPrima(vecchio, U1, "uno");
    riempiCollegate(vecchio, U1, "due");
    var nuovo = L.controllaImport(C, L.esportaJSON(C, vecchio)).stato;
    L.mostraSoluzione(C, nuovo, U1);
    var mem = memoriaFinta();
    mem.setItem(chiave, L.esportaJSON(C, nuovo));
    App.inizializza(P, S, E, null, opzioniProva(mem));
    verifica("ripresa: lo stato salvato nel browser si ricarica con esposizioni e blocchi",
             esposto(App.stato(), recordComune, U1.id) && !!App.stato().unita[U2.id].prima_fissata
             && App.esito().indexOf("Ripreso") === 0);
    mem.setItem(chiave, L.esportaJSON(C, vecchio));  // un'altra scheda rimasta indietro scrive uno stato vecchio
    App.salvaOra();
    var salvato = L.controllaImport(C, mem.getItem(chiave)).stato;
    verifica("ripresa: uno stato vecchio scritto da un'altra scheda non cancella l'esposizione al salvataggio",
             esposto(salvato, recordComune, U1.id) && !!salvato.unita[U2.id].prima_fissata);
    var mem2 = memoriaFinta();
    mem2.setItem(chiave, L.esportaJSON(C, vecchio));
    App.inizializza(P, S, E, null, opzioniProva(mem2));
    mem2.setItem(chiave, L.esportaJSON(C, nuovo));  // un'altra scheda mostra la soluzione
    App.salvaOra();
    verifica("ripresa: un'esposizione scritta da un'altra scheda si unisce allo stato in memoria",
             esposto(App.stato(), recordComune, U1.id) && !!App.stato().unita[U2.id].prima_fissata
             && esposto(L.controllaImport(C, mem2.getItem(chiave)).stato, recordComune, U1.id));
    var v1 = JSON.stringify(formato1(nuovo));
    var mem3 = memoriaFinta();
    mem3.setItem(chiave, v1);
    App.inizializza(P, S, E, null, opzioniProva(mem3));
    var esito3 = App.esito();
    App.salvaOra();
    var copie = Object.keys(mem3.dati).filter(function (k) { return k.indexOf(chiave + ":riserva:formato-1:") === 0; });
    var salvato3 = JSON.parse(mem3.getItem(chiave));
    verifica("ripresa dal formato 1: migrazione esplicita, copia del formato 1 conservata nel browser",
             esposto(App.stato(), recordComune, U1.id) && esito3.indexOf("formato 1") >= 0 && copie.length === 1
             && mem3.dati[copie[0]] === v1 && salvato3.schema === L.SCHEMA_STATO && salvato3.migrazioni.length === 1);
    var mem4 = memoriaFinta();
    mem4.setItem(chiave, "{non valido");
    App.inizializza(P, S, E, null, opzioniProva(mem4));
    var esito4 = App.esito();
    App.salvaOra();
    var riserve = Object.keys(mem4.dati).filter(function (k) { return k.indexOf(chiave + ":riserva:non-caricato:") === 0; });
    verifica("ripresa: uno stato salvato illeggibile non si perde (copia di riserva) e lo si segnala",
             riserve.length === 1 && mem4.dati[riserve[0]] === "{non valido" && esito4.indexOf("non è valido") >= 0);
    App.inizializza(P, S, E, L.nuovoStato(P), opzioniProva(null));
  });

  disegna();

  /* ------------------------------------------------------------ 11. immagini (asincrono) */
  if (simulato) {
    salta("immagini: tutte caricate dalla cartella", "DOM simulato: le immagini si caricano solo in un browser; " +
          "presenza e sha256 sono controllati da review_package.py verifica");
    disegna();
    return;
  }
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
