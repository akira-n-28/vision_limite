/* Interfaccia della revisione visiva offline di MAVIS (index.html). Legge window.PACCHETTO e SOLUZIONI (dati/*.js) e usa
   window.Logica (logica.js). Nessuna richiesta di rete: immagini e dati sono file locali della cartella. Deriva
   dall'interfaccia di revisioni/synth-v4-campione-2 (visore delle immagini, salvataggio, importazione), con una sola
   lista di schede.

   Primo passaggio: immagine originale, domanda, opzioni solo se appartengono alla domanda, risposta libera o «non
   determinabile», leggibilità, sufficienza, note. Target, annotazioni, esiti automatici, categorie e motivi di
   selezione restano in dati/soluzioni.js e si mostrano solo dopo il primo giudizio salvato.

   Salvataggio nel browser: prima di ogni scrittura lo stato salvato si rilegge e si unisce a quello in memoria
   (Logica.unisci, senza scelta), così un'esposizione registrata da un'altra scheda del browser o da una sessione
   precedente non si cancella mai. Uno stato salvato che non si riesce a leggere non si sovrascrive: si copia in una
   chiave di riserva. */
(function (root) {
  "use strict";
  var L = root.Logica;
  var PKG = null, CTX = null, SOL = null, STATO = null;
  var OPZ = {persisti: true};
  var vista = {pagina: "schede", id: null, filtro: "tutti", zoom: "1"};
  var cronometro = {id: null, inizio: 0};
  var voceElenco = {};
  var timerSalvataggio = null;
  var esitoSalvataggio = "";
  var ultimoScritto = null;
  var altraScheda = false;

  var NOMI = {
    corretta_pixel: "Il target candidato è giusto per i pixel?",
    corretta_pixel_diagnostica: "Annotazioni (tipo dichiarato e misure) coerenti con i pixel?",
    rispondibile: "Rispondibile dalla sola immagine e dalla domanda?",
    rispondibile_diagnostica: "Le classi vere si decidono dalla sola immagine?",
    scrittura_visibile: "Scrittura visibile uguale al visualizzato previsto, nella stessa forma?",
    prudenza_eccessiva: "Filtro troppo prudente?",
    giudizio_finale: "Giudizio finale"
  };
  var FASI = {
    non_iniziato: ["○", "non iniziato"],
    bozza: ["◔", "primo giudizio in bozza"],
    primo_giudizio: ["◑", "primo giudizio salvato, soluzione nascosta"],
    soluzione_vista: ["◕", "soluzione vista, giudizio finale incompleto"],
    completo: ["●", "completo"]
  };

  /* ------------------------------------------------------------ DOM */

  function $(id) {
    return document.getElementById(id);
  }

  function aggiungi(e, x) {
    if (x === null || x === undefined || x === false) {
      return;
    }
    if (Array.isArray(x)) {
      x.forEach(function (y) { aggiungi(e, y); });
    } else if (typeof x === "string" || typeof x === "number") {
      e.appendChild(document.createTextNode(String(x)));
    } else {
      e.appendChild(x);
    }
  }

  function el(tag, attrs) {
    var e = document.createElement(tag);
    var a = attrs || {};
    Object.keys(a).forEach(function (k) {
      var v = a[k];
      if (v === null || v === undefined || v === false) {
        return;
      }
      if (k === "testo") {
        e.textContent = v;
      } else if (k === "classe") {
        e.className = v;
      } else if (k.slice(0, 2) === "on") {
        e.addEventListener(k.slice(2), v);
      } else if (v === true) {
        e.setAttribute(k, "");
      } else {
        e.setAttribute(k, v);
      }
    });
    for (var i = 2; i < arguments.length; i++) {
      aggiungi(e, arguments[i]);
    }
    return e;
  }

  function quando(iso) {
    if (!iso) {
      return "—";
    }
    var d = new Date(iso);
    return d.toLocaleDateString("it-IT") + " " + d.toLocaleTimeString("it-IT");
  }

  function etichetta(nome, valore) {
    var x = L.VALORI[nome].filter(function (y) { return y[0] === valore; })[0];
    return x ? x[1] : valore;
  }

  function scelta(nome, valore, onchange, disabilitato) {
    var s = el("select", {disabled: !!disabilitato, "data-campo": nome});
    L.VALORI[nome].forEach(function (x) { s.appendChild(el("option", {value: x[0], testo: x[1]})); });
    s.value = valore;
    s.addEventListener("change", function () { onchange(s.value, s); });
    return s;
  }

  function areaTesto(valore, oninput, disabilitata, righe, campoNome) {
    var t = el("textarea", {rows: String(righe || 2), disabled: !!disabilitata, "data-campo": campoNome || null});
    t.value = valore;
    t.addEventListener("input", function () { oninput(t.value); });
    return t;
  }

  function campo(nome, aiuto, controllo) {
    return el("label", {classe: "campo"}, el("span", {classe: "etichetta", testo: nome}), controllo,
              aiuto ? el("span", {classe: "aiuto", testo: aiuto}) : null);
  }

  function messaggio(nodo, testi, tipo) {
    nodo.textContent = testi.join(" · ");
    nodo.className = "messaggio" + (tipo ? " " + tipo : "");
  }

  function conferma(testo) {
    return OPZ.conferma ? OPZ.conferma(testo) : root.confirm(testo);
  }

  /* ------------------------------------------------------------ immagini ingrandibili (come synth-v4-campione-2) */

  function figura(src, didascalia, conLente) {
    var img = el("img", {src: src, alt: didascalia || src, classe: "immagine"});
    var scatola = el("div", {classe: "scatola-immagine"}, img);
    var barra = el("div", {classe: "zoom"});
    var dimensioni = el("span", {classe: "dimensioni"});
    function applica(z) {
      vista.zoom = z;
      barra.querySelectorAll("button[data-z]").forEach(function (b) {
        b.classList.toggle("attivo", b.getAttribute("data-z") === z);
      });
      if (z === "adatta") {
        img.style.width = "";
        img.classList.add("adatta");
        img.classList.remove("nitida");
      } else {
        img.classList.remove("adatta");
        img.classList.toggle("nitida", Number(z) >= 2);
        if (img.naturalWidth) {
          img.style.width = (img.naturalWidth * Number(z)) + "px";
        }
      }
    }
    [["adatta", "adatta"], ["1", "100% (nativa)"], ["2", "200%"], ["3", "300%"], ["4", "400%"]].forEach(function (l) {
      barra.appendChild(el("button", {type: "button", "data-z": l[0], testo: l[1],
                                      onclick: function () { applica(l[0]); }}));
    });
    if (conLente !== false) {
      barra.appendChild(el("button", {type: "button", testo: "schermo intero",
                                      onclick: function () { apriLente(src, didascalia); }}));
    }
    img.addEventListener("load", function () {
      applica(vista.zoom);
      dimensioni.textContent = img.naturalWidth + " × " + img.naturalHeight + " px";
    });
    img.addEventListener("error", function () {
      scatola.appendChild(el("p", {classe: "errore", testo: "Immagine non trovata: " + src +
                                    ". Controlla di aver estratto tutta la cartella."}));
    });
    return el("figure", {classe: "figura"}, el("figcaption", {}, el("span", {testo: didascalia || ""}), " ", dimensioni),
              barra, scatola);
  }

  function apriLente(src, titolo) {
    var lente = $("lente");
    lente.innerHTML = "";
    lente.appendChild(el("div", {classe: "lente-barra"}, el("span", {testo: titolo || src}),
                         el("button", {type: "button", testo: "chiudi (Esc)", onclick: chiudiLente})));
    lente.appendChild(figura(src, titolo, false));
    lente.hidden = false;
  }

  function chiudiLente() {
    var lente = $("lente");
    if (lente) {
      lente.hidden = true;
      lente.innerHTML = "";
    }
  }

  /* ------------------------------------------------------------ salvataggio */

  function memoria() {
    if (!OPZ.persisti) {
      return null;
    }
    try {
      return OPZ.memoria || root.localStorage;
    } catch (e) {
      return null;
    }
  }

  function chiaveLocale() {
    return "revisione:" + PKG.id + ":v" + PKG.versione + ":" + PKG.impronta.slice(0, 16);
  }

  function marcaTempo() {
    return new Date().toISOString().slice(0, 19).replace(/:/g, "-");
  }

  /* Copia di riserva nel browser di un testo che non si carica: non si cancella mai. */
  function riserva(mem, motivo, testo) {
    var k = chiaveLocale() + ":riserva:" + motivo + ":" + marcaTempo();
    try {
      mem.setItem(k, testo);
      return k;
    } catch (e) {
      return null;
    }
  }

  function caricaLocale() {
    var mem = memoria();
    if (!mem) {
      return L.nuovoStato(PKG);
    }
    var testo = null;
    try {
      testo = mem.getItem(chiaveLocale());
    } catch (e) {
      esitoSalvataggio = "Salvataggio nel browser non disponibile: scarica spesso il file di avanzamento.";
      OPZ.persisti = false;
      return L.nuovoStato(PKG);
    }
    if (!testo) {
      esitoSalvataggio = "Nessun lavoro salvato in questo browser. Se hai già lavorato altrove, importa subito " +
                         "l'ultimo file di avanzamento (Riepilogo e file → Importa) prima di continuare.";
      return L.nuovoStato(PKG);
    }
    var r = L.controllaImport(CTX, testo);
    if (r.errori.length) {
      var k = riserva(mem, "non-caricato", testo);
      OPZ.bloccaScrittura = !k;  // senza copia di riserva lo stato salvato non si sovrascrive
      esitoSalvataggio = "Lo stato salvato nel browser non è valido e non è stato caricato (" +
                         r.errori.slice(0, 2).join("; ") + "). Non è stato cancellato" +
                         (k ? " (copia nel browser: " + k + ")" : " e non si sovrascrive") +
                         ": importa il tuo ultimo file di avanzamento.";
      return L.nuovoStato(PKG);
    }
    esitoSalvataggio = "Ripreso il lavoro salvato in questo browser (" + quando(r.stato.aggiornato_il) + ").";
    ultimoScritto = testo;
    L.applicaEsposizioni(CTX, r.stato);
    return r.stato;
  }

  /* Unisce allo stato in memoria quello salvato nel browser da un'altra scheda o finestra (conoscenza e voci nuove; il
     lavoro in conflitto resta quello in memoria). Restituisce true se lo stato è cambiato. */
  function sincronizza(mem) {
    var testo;
    try {
      testo = mem.getItem(chiaveLocale());
    } catch (e) {
      return false;
    }
    if (!testo || testo === ultimoScritto) {
      return false;
    }
    var r = L.controllaImport(CTX, testo);
    if (r.errori.length) {
      return false;
    }
    var prima = L.impronteConoscenza(STATO);
    var u = L.unisci(CTX, STATO, r.stato, "senza_scelta");
    var cambiato = u.nuove > 0 || u.esposizioni_nuove > 0 || u.archiviate > 0 ||
      L.impronteConoscenza(u.stato) !== prima;
    if (cambiato) {
      STATO = u.stato;
      altraScheda = true;
    }
    return cambiato;
  }

  function salvaLocale() {
    if (timerSalvataggio) {
      clearTimeout(timerSalvataggio);
      timerSalvataggio = null;
    }
    STATO.aggiornato_il = L.ora();
    var mem = memoria();
    if (!mem || OPZ.bloccaScrittura) {
      disegnaBarraSalvataggio();
      return;
    }
    var cambiato = sincronizza(mem);
    try {
      var testo = L.esportaJSON(CTX, STATO);
      mem.setItem(chiaveLocale(), testo);
      ultimoScritto = testo;
      esitoSalvataggio = "Salvato nel browser alle " + new Date().toLocaleTimeString("it-IT") + ".";
    } catch (e) {
      esitoSalvataggio = "Salvataggio nel browser non riuscito: scarica subito il file di avanzamento.";
    }
    if (cambiato) {
      disegna();
    } else {
      disegnaBarraSalvataggio();
    }
  }

  function salva() {
    if (timerSalvataggio) {
      clearTimeout(timerSalvataggio);
    }
    timerSalvataggio = setTimeout(salvaLocale, 300);
    STATO.aggiornato_il = L.ora();
  }

  function ultimaEsportazione() {
    var mem = memoria();
    try {
      return mem ? mem.getItem(chiaveLocale() + ":esportato") || "" : "";
    } catch (e) {
      return "";
    }
  }

  function segnaEsportazione() {
    var mem = memoria();
    try {
      if (mem) {
        mem.setItem(chiaveLocale() + ":esportato", STATO.aggiornato_il);
      }
    } catch (e) { /* solo un promemoria */ }
  }

  function scarica(nome, testo, tipo) {
    if (OPZ.scarica) {
      OPZ.scarica(nome, testo, tipo);
      return;
    }
    var blob = new Blob([testo], {type: tipo + ";charset=utf-8"});
    var url = URL.createObjectURL(blob);
    var a = el("a", {href: url, download: nome});
    document.body.appendChild(a);
    a.click();
    setTimeout(function () {
      URL.revokeObjectURL(url);
      a.parentNode.removeChild(a);
    }, 3000);
  }

  function scaricaAvanzamento() {
    fermaCronometro();
    salvaLocale();
    scarica(PKG.id + "_avanzamento_" + marcaTempo() + ".json", L.esportaJSON(CTX, STATO), "application/json");
    segnaEsportazione();
    avviaCronometro();
    disegnaBarraSalvataggio();
  }

  function disegnaBarraSalvataggio() {
    var barra = $("salvataggio");
    if (!barra) {
      return;
    }
    var esportato = ultimaEsportazione();
    var avviso = !esportato ? "File di avanzamento mai scaricato." :
      esportato !== STATO.aggiornato_il ? "Modifiche successive all'ultimo file scaricato." : "File scaricato aggiornato.";
    barra.innerHTML = "";
    barra.appendChild(el("span", {testo: esitoSalvataggio + " " + avviso}));
    if (altraScheda) {
      barra.appendChild(el("span", {classe: "errore", testo: "La revisione è aperta anche in un'altra scheda o " +
        "finestra: lavora in una sola. Le esposizioni registrate là sono state unite qui."}));
    }
    barra.appendChild(el("button", {type: "button", classe: "primario", testo: "Scarica il file di avanzamento",
                                    onclick: scaricaAvanzamento}));
  }

  /* ------------------------------------------------------------ tempo per scheda */

  function avviaCronometro() {
    cronometro.id = vista.id;
    cronometro.inizio = vista.id && vista.pagina === "schede" ? Date.now() : 0;
  }

  function fermaCronometro() {
    if (cronometro.id && cronometro.inizio) {
      var ms = Math.min(Date.now() - cronometro.inizio, 30 * 60 * 1000);
      if (ms > 2000 && CTX.idx[cronometro.id]) {
        L.aggiungiTempo(STATO, CTX.idx[cronometro.id], ms);
        salva();
      }
    }
    cronometro.inizio = 0;
  }

  /* ------------------------------------------------------------ navigazione */

  function vai(id) {
    if (!CTX.idx[id]) {
      return;
    }
    fermaCronometro();
    chiudiLente();
    vista.pagina = "schede";
    vista.id = id;
    STATO.posizione = id;
    if (root.location.hash !== "#" + id) {
      root.location.hash = id;
    }
    disegna();
    root.scrollTo(0, 0);
    avviaCronometro();
  }

  function apriPagina(pagina) {
    fermaCronometro();
    vista.pagina = pagina;
    if (pagina === "schede" && !vista.id) {
      vista.id = CTX.ordine.filter(function (id) { return faseDi(id).fase !== "completo"; })[0] || CTX.ordine[0];
    }
    disegna();
    avviaCronometro();
  }

  function sposta(delta) {
    var i = CTX.ordine.indexOf(vista.id);
    var j = i + delta;
    if (j >= 0 && j < CTX.ordine.length) {
      vai(CTX.ordine[j]);
    }
  }

  function prossimoDaCompletare() {
    var i = CTX.ordine.indexOf(vista.id);
    for (var n = 1; n <= CTX.ordine.length; n++) {
      var id = CTX.ordine[(i + n) % CTX.ordine.length];
      if (faseDi(id).fase !== "completo") {
        vai(id);
        return true;
      }
    }
    return false;
  }

  function daHash() {
    var id = decodeURIComponent(root.location.hash.slice(1));
    if (id && id !== vista.id && CTX.idx[id]) {
      vai(id);
    }
  }

  function faseDi(id) {
    return L.statoScheda(CTX, CTX.idx[id], STATO.schede[id]);
  }

  /* ------------------------------------------------------------ testata ed elenco */

  function disegnaTestata() {
    var nav = $("pagine");
    if (!nav) {
      return;
    }
    var r = L.riepilogo(CTX, STATO);
    nav.innerHTML = "";
    [["schede", "Schede " + r.completo + "/" + r.totale + (r.incerto ? " · " + r.incerto + " incerte" : "")],
     ["riepilogo", "Riepilogo e file"]].forEach(function (p) {
      var b = el("button", {type: "button", classe: "scheda" + (vista.pagina === p[0] ? " attiva" : ""), testo: p[1],
                            onclick: function () { apriPagina(p[0]); }});
      if (p[0] === "schede") {
        var barra = el("span", {classe: "avanzamento"});
        barra.appendChild(el("span", {classe: "pieno", style: "width:" + (100 * r.completo / r.totale) + "%"}));
        b.appendChild(barra);
      }
      nav.appendChild(b);
    });
  }

  function segno(id) {
    var s = faseDi(id);
    var f = FASI[s.fase] || ["?", s.fase];
    return {testo: f[0] + (s.incerto ? "?" : "") + (s.non_giudicabile ? "!" : "") + (s.non_indipendente ? "~" : ""),
            titolo: f[1] + (s.incerto ? ", con valori incerti" : "") +
              (s.non_giudicabile ? ", con valori non giudicabili" : "") +
              (s.non_indipendente ? ", primo giudizio non indipendente" : ""),
            fase: s.fase, incerto: s.incerto || s.non_giudicabile};
  }

  function passaFiltro(id) {
    var s = segno(id);
    return vista.filtro === "tutti" || (vista.filtro === "da_completare" && s.fase !== "completo") ||
      (vista.filtro === "incerti" && s.incerto) || (vista.filtro === "completi" && s.fase === "completo");
  }

  function disegnaElenco() {
    var aside = $("elenco");
    if (!aside) {
      return;
    }
    aside.innerHTML = "";
    voceElenco = {};
    if (vista.pagina !== "schede") {
      aside.hidden = true;
      return;
    }
    aside.hidden = false;
    var filtro = el("select", {});
    [["tutti", "tutte"], ["da_completare", "da completare"], ["incerti", "incerte o non giudicabili"],
     ["completi", "complete"]].forEach(function (x) { filtro.appendChild(el("option", {value: x[0], testo: x[1]})); });
    filtro.value = vista.filtro;
    filtro.addEventListener("change", function () {
      vista.filtro = filtro.value;
      disegnaElenco();
    });
    aside.appendChild(el("div", {classe: "filtro"}, "Mostra ", filtro));
    aside.appendChild(el("p", {classe: "legenda", testo: "○ non iniziata · ◔ bozza · ◑ primo giudizio salvato · " +
                                                     "◕ soluzione vista · ● completa · ? incerta · ! non giudicabile · " +
                                                     "~ primo giudizio non indipendente"}));
    var lista = el("ul", {});
    CTX.ordine.forEach(function (id) {
      if (!passaFiltro(id) && id !== vista.id) {
        return;
      }
      var s = segno(id);
      var b = el("button", {type: "button", classe: "voce" + (id === vista.id ? " corrente" : ""), title: s.titolo,
                            onclick: function () { vai(id); }},
                 el("span", {classe: "segno", testo: s.testo}), " " + id + " · " + CTX.idx[id].record);
      voceElenco[id] = b;
      lista.appendChild(el("li", {}, b));
    });
    aside.appendChild(lista);
    var corrente = voceElenco[vista.id];
    if (corrente && corrente.scrollIntoView) {
      corrente.scrollIntoView({block: "nearest"});
    }
  }

  function aggiornaSegno(id) {
    var b = voceElenco[id];
    if (b) {
      var s = segno(id);
      b.firstChild.textContent = s.testo;
      b.title = s.titolo;
    }
    disegnaTestata();
  }

  function navigazione(id) {
    var i = CTX.ordine.indexOf(id);
    var msg = el("span", {classe: "messaggio"});
    return el("div", {classe: "navigazione"},
              el("button", {type: "button", disabled: i <= 0, testo: "← precedente", onclick: function () { sposta(-1); }}),
              el("span", {classe: "posizione", testo: (i + 1) + " di " + CTX.ordine.length}),
              el("button", {type: "button", disabled: i >= CTX.ordine.length - 1, testo: "successiva →",
                            onclick: function () { sposta(1); }}),
              el("button", {type: "button", testo: "prossima da completare", onclick: function () {
                if (!prossimoDaCompletare()) {
                  messaggio(msg, ["Tutte le schede sono complete."], "ok");
                }
              }}), msg);
  }

  /* ------------------------------------------------------------ scheda: primo passaggio */

  function bloccoDomanda(s) {
    var box = el("div", {classe: "blocco-domanda"});
    if (s.richiesta === "diagnostica") {
      box.appendChild(el("p", {classe: "diagnostica"}, el("strong", {testo: "Richiesta diagnostica"}),
                         " · non è una domanda del training e non ha una golden answer."));
    }
    box.appendChild(el("h3", {testo: s.richiesta === "diagnostica" ? "Richiesta" : "Domanda"}));
    box.appendChild(el("pre", {classe: "domanda", testo: s.domanda}));
    if (s.opzioni.length) {
      box.appendChild(el("h3", {testo: "Opzioni della domanda"}));
      box.appendChild(el("ul", {classe: "opzioni"}, s.opzioni.map(function (o) { return el("li", {testo: o}); })));
    }
    return box;
  }

  function metaScheda(s) {
    var righe = [["scheda", s.id + " · " + s.posizione + " di " + CTX.ordine.length], ["record", s.record],
                 ["immagine", "caption.jpg originale · " + s.larghezza + " × " + s.altezza + " px · sha256 " +
                  s.immagine_sha256.slice(0, 16) + "…"]];
    return el("table", {classe: "meta"}, righe.map(function (r) {
      return el("tr", {}, el("th", {testo: r[0]}), el("td", {testo: r[1]}));
    }));
  }

  function statoCollegata(id) {
    var w = STATO.schede[id];
    if (w && w.prima_fissata) {
      return "primo giudizio fissato";
    }
    if (w && w.esposta_prima) {
      return "già esposta: il suo primo giudizio non sarà indipendente";
    }
    if (!w || !w.prima) {
      return "manca il primo giudizio";
    }
    return L.bozzaUgualeAPrima(w) ? "primo giudizio salvato" : "modifiche non salvate al primo giudizio";
  }

  function bloccoCollegate(s) {
    var coll = CTX.collegati[s.id] || [];
    if (!coll.length) {
      return null;
    }
    var b = L.bloccanti(CTX, STATO, s);
    return el("div", {classe: "collegate"}, el("h4", {testo: "Schede collegate (" + coll.length + ")"}),
              el("p", {classe: "aiuto", testo: L.AIUTI.collegate}),
              el("ul", {}, coll.map(function (c) {
                var manca = b.mancanti.indexOf(c.id) >= 0 || b.non_salvate.indexOf(c.id) >= 0;
                return el("li", {classe: manca ? "manca" : ""},
                          el("button", {type: "button", classe: "collegamento", "data-id": c.id, testo: c.id,
                                        onclick: function () { vai(c.id); }}), " · " + statoCollegata(c.id));
              })));
  }

  function sezionePrima(s) {
    var v = L.voce(STATO, s, false);
    var fissato = !!(v && v.prima_fissata);
    var mostrata = !!(v && v.soluzione_mostrata_il);
    var b = fissato ? v.prima.bozza : (v ? v.bozza : {risposta: "", non_determinabile: false, leggibilita: L.NON,
                                                      sufficienza: L.NON, note: ""});
    var box = el("section", {classe: "passaggio prima"}, el("h3", {testo: "1. Primo giudizio, senza la soluzione"}));
    if (s.controllo_noto) {
      box.appendChild(el("p", {classe: "avviso controllo"}, el("strong", {testo: "Controllo noto. "}),
        "Il caso è già descritto nei rapporti di C9: il primo giudizio di questa scheda non è indipendente e si " +
        "registra come tale."));
    }
    if (v && v.esposta_prima) {
      box.appendChild(el("p", {classe: "avviso"}, el("strong", {testo: "Primo giudizio non indipendente. "}),
        "Il record o il gruppo di questa scheda è stato esposto il " + quando(v.esposta_prima.quando) + " (scheda " +
        v.esposta_prima.tramite + ") prima che il primo giudizio fosse salvato."));
    }
    var msg = el("p", {classe: "messaggio"});
    var rivela = el("button", {type: "button", classe: "rivela", testo: "Mostra soluzione e materiale di confronto"});
    var motivo = el("span", {classe: "aiuto"});
    function aggiornaRivela() {
      var p = L.puoMostrare(CTX, STATO, s);
      rivela.disabled = !p.ok;
      motivo.textContent = p.ok ? "Dopo, il primo giudizio resta fissato e non si modifica più." : p.motivo;
    }
    function modifica(nome, valore) {
      var errori = L.aggiornaBozza(STATO, s, nome, valore);
      if (errori.length) {
        messaggio(msg, errori, "errore");
      }
      salva();
      aggiornaRivela();
      aggiornaSegno(s.id);
    }
    var spunta = el("input", {type: "checkbox", disabled: fissato, "data-campo": "non_determinabile"});
    spunta.checked = b.non_determinabile;
    spunta.addEventListener("change", function () { modifica("non_determinabile", spunta.checked); });
    box.appendChild(campo("Risposta", L.AIUTI.risposta, areaTesto(b.risposta, function (t) { modifica("risposta", t); },
                                                                  fissato, s.richiesta === "diagnostica" ? 2 : 3,
                                                                  "risposta")));
    box.appendChild(el("label", {classe: "campo spunta"}, spunta, el("span", {testo: " non determinabile"}),
                       el("span", {classe: "aiuto", testo: L.AIUTI.non_determinabile})));
    box.appendChild(campo("Leggibilità", L.AIUTI.leggibilita,
                          scelta("leggibilita", b.leggibilita, function (x) { modifica("leggibilita", x); }, fissato)));
    box.appendChild(campo("Sufficienza delle informazioni", L.AIUTI.sufficienza,
                          scelta("sufficienza", b.sufficienza, function (x) { modifica("sufficienza", x); }, fissato)));
    box.appendChild(campo("Note (facoltative)", L.AIUTI.note_prima,
                          areaTesto(b.note, function (t) { modifica("note", t); }, fissato, 2, "note")));
    if (fissato) {
      box.appendChild(el("p", {classe: "fissato", testo: "Primo giudizio salvato il " + quando(v.prima.salvata_il) +
        " e fissato il " + quando(v.prima_fissata.quando) + ". Non si modifica più: i ripensamenti vanno nel " +
        "giudizio finale, anche nel campo «risposta dopo la soluzione»."}));
    }
    if (mostrata) {
      aggiungi(box, bloccoCollegate(s));
      return box;
    }
    if (!fissato) {
      box.appendChild(el("div", {classe: "azioni"},
        el("button", {type: "button", classe: "primario salva-prima", testo: "Salva il primo giudizio",
                      onclick: function () {
                        var errori = L.salvaPrima(STATO, s);
                        salvaLocale();
                        if (errori.length) {
                          messaggio(msg, errori, "errore");
                          aggiornaRivela();
                          aggiornaSegno(s.id);
                        } else {
                          disegna();
                        }
                      }})));
    }
    box.appendChild(msg);
    aggiungi(box, bloccoCollegate(s));
    rivela.addEventListener("click", function () {
      if (!conferma("Dopo aver visto la soluzione il primo giudizio resta fissato e non si modifica più. Mostrare la " +
                    "soluzione?")) {
        return;
      }
      var errori = L.mostraSoluzione(CTX, STATO, s);
      salvaLocale();
      if (errori.length) {
        messaggio(msg, errori, "errore");
      } else {
        disegna();
      }
    });
    aggiornaRivela();
    box.appendChild(el("div", {classe: "azioni"}, rivela, " ", motivo));
    return box;
  }

  /* ------------------------------------------------------------ scheda: soluzione e giudizio finale */

  function dettaglio(titolo, oggetto) {
    return el("details", {classe: "json"}, el("summary", {testo: titolo}),
              el("pre", {testo: typeof oggetto === "string" ? oggetto : JSON.stringify(oggetto, null, 1)}));
  }

  function tabella(righe) {
    return el("table", {classe: "soluzioni"}, righe.filter(function (r) { return r; }).map(function (r) {
      return el("tr", {}, el("th", {testo: r[0]}), el("td", {}, r[1]));
    }));
  }

  function testo(x) {
    if (x === null || x === undefined) {
      return "—";
    }
    return typeof x === "string" ? x : JSON.stringify(x);
  }

  function materialeSoluzione(s, x) {
    var box = el("div", {classe: "soluzione"});
    box.appendChild(el("p", {classe: "nota", testo: "Materiale di confronto: un target o un'annotazione automatica " +
      "non è giusta per definizione, e l'accordo con la tua risposta non la rende «corretta» né «gold»."}));
    var righe = [
      ["target candidato", x.target === null ? el("em", {testo: x.nota_target || "nessuna golden answer"}) :
        el("pre", {classe: "target", testo: x.target})],
      x.nota_target && x.target !== null ? ["nota sul target", x.nota_target] : null,
      ["categoria del campione", x.categoria + " / " + x.sottocategoria + " · cella " + x.cella],
      ["motivo della selezione", x.motivo_selezione],
      ["regola dello strato", x.regola],
      ["tipo di richiesta", x.qa_candidata ? "QA candidata del pool C9 (" + testo(x.compito) + ", stato del compito " +
        testo(x.stato_compito) + ")" : "richiesta diagnostica, non una QA"],
      x.qa_sha256 ? ["sha256 della QA", x.qa_sha256] : null,
      x.nota_controllo ? ["controllo noto", x.nota_controllo] : null
    ];
    box.appendChild(tabella(righe));
    if (x.valore) {
      box.appendChild(el("h4", {testo: "Valore della misura " + x.valore.elemento}));
      box.appendChild(tabella([["esatto (annotazioni)", x.valore.esatto],
                               ["visualizzato previsto (regola del renderer)", x.valore.visualizzato_previsto],
                               ["rappresentazione registrata", x.valore.rappresentazione],
                               ["risposta di lettura (target)", x.valore.risposta_lettura]]));
      box.appendChild(el("p", {classe: "nota", testo: "Il visualizzato previsto viene dalla regola del renderer, non " +
        "dai pixel: confrontalo con ciò che vedi. Le equivalenze matematiche (per esempio √841 = 29) vanno nel campo " +
        "apposito e non cambiano la trascrizione."}));
    }
    if (x.c9) {
      box.appendChild(el("h4", {testo: "Esito C9 della figura " + x.figura}));
      box.appendChild(tabella([
        ["esito", x.c9.esito + (x.c9.motivo ? " · motivo " + x.c9.motivo : "")],
        x.c9.opzioni ? ["opzioni (ordine della domanda)", x.c9.opzioni.join(" / ")] : null,
        x.c9.risposta ? ["risposta C9", x.c9.risposta] : null,
        x.c9.confutazioni ? ["confutazioni delle altre opzioni (misure stampate)", testo(x.c9.confutazioni)] : null,
        x.c9.vere ? ["opzioni vere secondo le annotazioni", testo(x.c9.vere)] : null,
        x.c9.non_confutate ? ["opzioni false non confutate dalle misure stampate", testo(x.c9.non_confutate)] : null,
        x.c9.ignote ? ["verità ignote", testo(x.c9.ignote)] : null,
        x.c9.dettaglio ? ["dettaglio", testo(x.c9.dettaglio)] : null,
        ["verità delle opzioni secondo le annotazioni (non gold)", testo(x.verita_annotazioni)],
        ["opzioni confutate dalle sole misure stampate", testo(x.confutazioni_stampate)],
        x.confutazioni_con_segno_retto ? ["confutate con il segno dell'angolo retto (convenzione da verificare)",
                                          testo(x.confutazioni_con_segno_retto)] : null,
        ["tipo dichiarato e misure della figura", testo(x.fatti)],
        x.storica ? ["domanda storica (in quarantena)", x.storica.domanda + " → " + x.storica.risposta] : null
      ]));
    }
    box.appendChild(el("h4", {testo: "Annotazioni del record"}));
    box.appendChild(el("p", {classe: "didascalia", testo: x.annotazioni.didascalia}));
    box.appendChild(dettaglio("diagram_generation", x.annotazioni.diagram_generation));
    box.appendChild(dettaglio("Valori del record (" + x.valori_del_record.length + " misure)", x.valori_del_record));
    box.appendChild(dettaglio("Record nel manifest: split, stato, gruppo, profondità, affollamento", x.record));
    box.appendChild(dettaglio("Immagine originale e selezione", {immagine: x.immagine, selezione: x.selezione}));
    return box;
  }

  function sezioneDopo(s) {
    var v = L.voce(STATO, s, false);
    if (!v || !v.soluzione_mostrata_il) {
      return el("section", {classe: "passaggio dopo nascosto"},
                el("h3", {testo: "2. Soluzione, annotazioni e giudizio finale"}),
                el("p", {classe: "nota", testo: "Si apre dopo aver salvato il primo giudizio e premuto «Mostra " +
                  "soluzione». Fino ad allora restano nascosti target, annotazioni, esiti automatici, categoria e " +
                  "motivi della selezione."}));
    }
    var x = SOL[s.id];
    var m = L.modulo(CTX, s.id);
    var box = el("section", {classe: "passaggio dopo soluzione"},
                 el("h3", {testo: "2. Soluzione, annotazioni e giudizio finale"}));
    box.appendChild(el("p", {classe: "nota", testo: "Soluzione mostrata il " + quando(v.soluzione_mostrata_il) + "."}));
    var p = v.prima.bozza;
    box.appendChild(el("p", {classe: "ricordo"},
                       L.indipendente(CTX, s, v) ? "Risposta iniziale (indipendente): " : "Risposta iniziale (NON indipendente): ",
                       el("strong", {testo: p.non_determinabile ? "non determinabile" : (p.risposta || "—")}),
                       " · leggibilità: " + etichetta("leggibilita", p.leggibilita) + " · sufficienza: " +
                       etichetta("sufficienza", p.sufficienza)));
    box.appendChild(materialeSoluzione(s, x));
    var stato = el("div", {classe: "stato-caso"});
    function aggiornaStato(errori) {
      var st = L.statoScheda(CTX, s, L.voce(STATO, s, false));
      var manca = L.mancantiDopo(CTX, s, L.voce(STATO, s, false));
      stato.innerHTML = "";
      stato.appendChild(el("strong", {testo: FASI[st.fase][1]}));
      if (manca.length) {
        stato.appendChild(el("span", {testo: " · mancano: " + manca.join(", ")}));
      }
      if (errori && errori.length) {
        stato.appendChild(el("p", {classe: "errore", testo: errori.join(" · ")}));
      }
      aggiornaSegno(s.id);
    }
    function cambia(c, valore) {
      var errori = L.aggiornaDopo(CTX, STATO, s, c, valore);
      salva();
      aggiornaStato(errori);
    }
    var d = v.dopo;
    var fs = el("fieldset", {classe: "giudizio"}, el("legend", {testo: "Giudizio finale"}));
    m.scelte.forEach(function (c) {
      var nome = m.diagnostica && NOMI[c + "_diagnostica"] ? NOMI[c + "_diagnostica"] : NOMI[c];
      var aiuto = m.diagnostica && L.AIUTI[c + "_diagnostica"] ? L.AIUTI[c + "_diagnostica"] : L.AIUTI[c];
      fs.appendChild(campo(nome, aiuto, scelta(c, d[c], function (val) { cambia(c, val); })));
    });
    if (m.lettura) {
      fs.appendChild(campo("Equivalenze matematiche (facoltative, a parte)", L.AIUTI.equivalenza,
                           areaTesto(d.equivalenza, function (t) { cambia("equivalenza", t); }, false, 1, "equivalenza")));
    }
    if (m.diagnostica) {
      fs.appendChild(campo("Classi che ritieni vere (dopo le annotazioni)", L.AIUTI.risposte_vere,
                           areaTesto(d.risposte_vere, function (t) { cambia("risposte_vere", t); }, false, 1,
                                     "risposte_vere")));
    }
    var difetto = el("input", {type: "text", list: "tipi-difetto", "data-campo": "tipo_difetto"});
    difetto.value = d.tipo_difetto;
    difetto.addEventListener("input", function () { cambia("tipo_difetto", difetto.value); });
    fs.appendChild(campo("Tipo di difetto", L.AIUTI.tipo_difetto, difetto));
    fs.appendChild(campo("Note del giudizio finale", L.AIUTI.note_dopo,
                         areaTesto(d.note, function (t) { cambia("note", t); }, false, 3, "note_dopo")));
    fs.appendChild(campo("Risposta dopo la soluzione (non indipendente)", L.AIUTI.risposta_dopo_soluzione,
                         areaTesto(d.risposta_dopo_soluzione, function (t) { cambia("risposta_dopo_soluzione", t); },
                                   false, 2, "risposta_dopo_soluzione")));
    box.appendChild(fs);
    box.appendChild(stato);
    aggiornaStato([]);
    return box;
  }

  function costruisciVistaScheda(s) {
    var nodo = el("div", {classe: "caso", "data-id": s.id});
    nodo.appendChild(el("h2", {testo: "Scheda " + s.id}));
    nodo.appendChild(navigazione(s.id));
    nodo.appendChild(el("p", {classe: "consegna", testo: s.consegna}));
    nodo.appendChild(metaScheda(s));
    nodo.appendChild(el("div", {classe: "galleria"}, figura(s.immagine, s.record + " · caption.jpg")));
    nodo.appendChild(bloccoDomanda(s));
    nodo.appendChild(sezionePrima(s));
    nodo.appendChild(sezioneDopo(s));
    return nodo;
  }

  /* ------------------------------------------------------------ riepilogo, esportazione, importazione */

  function tabellaRiepilogo(r) {
    var colonne = [["totale", "totale"], ["completo", "complete"], ["soluzione_vista", "soluzione vista"],
                   ["primo_giudizio", "primo giudizio salvato"], ["bozza", "in bozza"], ["non_iniziato", "non iniziate"],
                   ["incerto", "con valori incerti"], ["non_giudicabile", "con valori non giudicabili"],
                   ["non_indipendente", "primo giudizio non indipendente"]];
    return el("table", {classe: "riepilogo"},
              el("tr", {}, colonne.map(function (c) { return el("th", {testo: c[1]}); })),
              el("tr", {classe: "totale"}, colonne.map(function (c) { return el("td", {testo: String(r[c[0]])}); })));
  }

  function tabellaEsposizioni() {
    if (!STATO.esposizioni.length) {
      return el("p", {classe: "nota", testo: "Nessun record esposto finora."});
    }
    var t = el("table", {classe: "esposizioni"}, el("tr", {}, ["quando", "record o gruppo esposto", "tramite la scheda"]
      .map(function (h) { return el("th", {testo: h}); })));
    STATO.esposizioni.slice().sort(function (a, b) { return a.quando < b.quando ? -1 : 1; }).forEach(function (e) {
      t.appendChild(el("tr", {}, [quando(e.quando), e.chiave.length > 40 ? e.chiave.slice(0, 22) + "…" : e.chiave,
                                  e.tramite].map(function (c) { return el("td", {testo: c}); })));
    });
    return t;
  }

  function costruisciRiepilogo() {
    var r = L.riepilogo(CTX, STATO);
    var nodo = el("div", {classe: "caso riepilogo"});
    nodo.appendChild(el("h2", {testo: "Riepilogo, salvataggio e restituzione"}));
    nodo.appendChild(tabellaRiepilogo(r));
    nodo.appendChild(el("p", {classe: "nota", testo: "Campione diagnostico, non rappresentativo della frequenza degli " +
      "errori: i conteggi servono a seguire il lavoro, non a stimare tassi."}));
    nodo.appendChild(el("h3", {testo: "Record esposti (" + r.esposizioni + ") e archivio"}));
    nodo.appendChild(el("p", {classe: "nota", testo: "Ogni volta che mostri una soluzione si registrano record e gruppo " +
      "esposti, con data e scheda. Il registro non si cancella, nemmeno importando un file più vecchio. Versioni " +
      "scartate da importazioni, conservate nell'archivio del file di avanzamento: " + r.archivio + "."}));
    nodo.appendChild(tabellaEsposizioni());
    var nome = el("input", {type: "text"});
    nome.value = STATO.revisore;
    nome.addEventListener("input", function () {
      STATO.revisore = nome.value;
      salva();
    });
    nodo.appendChild(campo("Revisore (facoltativo)", "Il nome o le iniziali registrate nel file di avanzamento.", nome));
    var msg = el("p", {classe: "messaggio"});
    nodo.appendChild(el("h3", {testo: "File da scaricare"}));
    nodo.appendChild(el("div", {classe: "azioni"},
      el("button", {type: "button", classe: "primario", testo: "Scarica il file di avanzamento (JSON)",
                    onclick: scaricaAvanzamento}),
      el("button", {type: "button", classe: "scarica-tsv", testo: "Scarica esiti.tsv", onclick: function () {
        scarica("esiti_" + marcaTempo() + ".tsv", L.tsvEsiti(CTX, STATO), "text/tab-separated-values");
      }})));
    nodo.appendChild(el("p", {classe: "nota", testo: "Il file JSON contiene tutto il lavoro (primo giudizio, giudizio " +
      "finale, record esposti, archivio, tempi ed eventi): è la copia da conservare e da restituire. Nel TSV gli a capo " +
      "sono \\n, le tabulazioni \\t e la barra rovesciata \\\\; un campo non compilato vale «non_compilato», uno che " +
      "non riguarda la scheda «non_applicabile»."}));
    var copia = el("textarea", {rows: "6", readonly: true, classe: "copia"});
    nodo.appendChild(el("details", {}, el("summary", {testo: "Se lo scaricamento non funziona: copia il JSON a mano"}),
      el("p", {classe: "nota", testo: "Premi il pulsante, seleziona tutto il testo e incollalo in un file di testo " +
                                      "con estensione .json."}),
      el("button", {type: "button", testo: "Mostra il JSON", onclick: function () {
        copia.value = L.esportaJSON(CTX, STATO);
        copia.select();
      }}), copia));
    nodo.appendChild(el("h3", {testo: "Riprendere da un file di avanzamento"}));
    var input = el("input", {type: "file", accept: ".json,application/json"});
    var conflitti = el("div", {classe: "conflitti"});
    input.addEventListener("change", function () {
      var f = input.files && input.files[0];
      if (!f) {
        return;
      }
      var lettore = new FileReader();
      lettore.onload = function () { importaTesto(String(lettore.result), msg, conflitti); };
      lettore.onerror = function () { messaggio(msg, ["File non leggibile."], "errore"); };
      lettore.readAsText(f, "utf-8");
      input.value = "";
    });
    nodo.appendChild(campo("Importa il file JSON scaricato in una sessione precedente", "Si controllano pacchetto, " +
      "versione, impronta e identificativi. Le esposizioni e i primi giudizi fissati non si perdono mai; il lavoro in " +
      "conflitto non si sovrascrive in silenzio e le versioni scartate restano nell'archivio.", input));
    nodo.appendChild(msg);
    nodo.appendChild(conflitti);
    nodo.appendChild(el("h3", {testo: "Che cosa ricaricare su GitHub"}));
    nodo.appendChild(el("p", {testo: "Nella cartella risultati/ di questo pacchetto: l'ultimo file di avanzamento " +
      "(…_avanzamento_….json) e, se vuoi, l'ultimo esiti_….tsv. Non modificare gli altri file del pacchetto. Le " +
      "istruzioni complete sono in LEGGIMI.md."}));
    nodo.appendChild(el("p", {classe: "nota", testo: "Pacchetto " + PKG.id + " v" + PKG.versione + " · impronta " +
      PKG.impronta + " · " + PKG.sorgente.manifest + " (manifest.json sha256 " + PKG.sorgente.manifest_sha256 + ")"}));
    return nodo;
  }

  function sintesiUnione(u) {
    var parti = [u.nuove + " schede aggiunte"];
    if (u.esposizioni_nuove) {
      parti.push(u.esposizioni_nuove + " esposizioni aggiunte");
    }
    if (u.decise_da_fissazione.length) {
      parti.push("vale il primo giudizio fissato per " + u.decise_da_fissazione.join(", "));
    }
    if (u.archiviate) {
      parti.push(u.archiviate + " versioni scartate conservate nell'archivio");
    }
    return parti.join("; ");
  }

  /* Importa un file di avanzamento. Prima scarica una copia dello stato del browser (se c'è lavoro), poi unisce subito
     la conoscenza del file (esposizioni, primi giudizi fissati, schede nuove); il lavoro in conflitto resta quello del
     browser finché non scegli. Restituisce l'esito dell'unione, o null se il file è rifiutato. */
  function importaTesto(testoFile, msg, box) {
    msg = msg || el("p", {});
    box = box || el("div", {});
    box.innerHTML = "";
    var r = L.controllaImport(CTX, testoFile);
    if (r.errori.length) {
      messaggio(msg, ["File rifiutato: " + r.errori.slice(0, 8).join("; ") +
                      (r.errori.length > 8 ? " e altri " + (r.errori.length - 8) : "")], "errore");
      return null;
    }
    var vuoto = !Object.keys(STATO.schede).length && !STATO.esposizioni.length;
    if (!vuoto) {
      scarica(PKG.id + "_copia_del_browser_" + marcaTempo() + ".json", L.esportaJSON(CTX, STATO), "application/json");
    }
    var u = L.unisci(CTX, STATO, r.stato, "senza_scelta");
    STATO = u.stato;
    OPZ.bloccaScrittura = false;
    salvaLocale();
    var testa = vuoto ? "" : "Copia del browser scaricata prima dell'importazione. ";
    if (!u.conflitti.length) {
      messaggio(msg, [testa + "File importato: " + sintesiUnione(u) + "."], "ok");
      disegna();
      return u;
    }
    messaggio(msg, [testa + "Importate le schede nuove e la conoscenza del file (" + sintesiUnione(u) + "). " +
                    u.conflitti.length + " schede hanno lavoro diverso nel browser e nel file e per ora restano come " +
                    "nel browser: scegli qui sotto."], "errore");
    box.appendChild(el("ul", {}, u.conflitti.slice(0, 80).map(function (id) {
      return el("li", {testo: id + ": nel browser modificata il " + quando(STATO.schede[id].aggiornato_il) +
                              ", nel file il " + quando(r.stato.schede[id].aggiornato_il)});
    })));
    function scegli(modo, testoFatto) {
      return function () {
        var w = L.unisci(CTX, STATO, r.stato, modo);
        STATO = w.stato;
        salvaLocale();
        box.innerHTML = "";
        messaggio(msg, [testoFatto + " " + sintesiUnione(w) + ". Esposizioni e primi giudizi fissati restano."], "ok");
        disegna();
      };
    }
    box.appendChild(el("div", {classe: "azioni"},
      el("button", {type: "button", classe: "tieni-browser", testo: "Tieni le versioni del browser (quelle del file " +
                    "vanno nell'archivio)", onclick: scegli("browser", "Tenute le versioni del browser.")}),
      el("button", {type: "button", classe: "usa-file", testo: "Usa le versioni del file per le schede in conflitto",
                    onclick: scegli("file", "Usate le versioni del file per le schede in conflitto.")}),
      el("button", {type: "button", classe: "sostituisci", testo: "Sostituisci tutto con il file (tranne esposizioni " +
                    "e primi giudizi fissati)", onclick: scegli("sostituisci", "Stato sostituito con il file, " +
                    "conservando esposizioni e primi giudizi fissati.")})));
    return u;
  }

  /* ------------------------------------------------------------ disegno */

  function disegna() {
    disegnaTestata();
    disegnaElenco();
    disegnaBarraSalvataggio();
    var main = $("caso");
    if (!main) {
      return;
    }
    main.innerHTML = "";
    if (vista.pagina === "riepilogo") {
      main.appendChild(costruisciRiepilogo());
    } else if (CTX.idx[vista.id]) {
      main.appendChild(costruisciVistaScheda(CTX.idx[vista.id]));
    }
  }

  /* opzioni: persisti (salvataggio nel browser), memoria (oggetto con getItem e setItem al posto di localStorage),
     scarica(nome, testo, tipo) e conferma(testo) al posto dello scaricamento e della finestra di conferma del browser:
     l'autotest le usa per non toccare né il lavoro salvato né la cartella Download. */
  function inizializza(pkg, sol, stato, opzioni) {
    PKG = pkg;
    SOL = sol;
    CTX = L.contesto(pkg, sol);
    OPZ = opzioni || {persisti: true};
    ultimoScritto = null;
    altraScheda = false;
    esitoSalvataggio = "";
    STATO = stato || caricaLocale();
    if (stato) {
      L.applicaEsposizioni(CTX, STATO);
    }
  }

  function avvia() {
    if (!root.PACCHETTO || !root.SOLUZIONI || !root.Logica) {
      document.body.appendChild(el("p", {classe: "errore", testo: "Dati del pacchetto non caricati. Estrai tutta la " +
        "cartella (con dati/ e immagini/) e apri index.html da lì."}));
      return;
    }
    inizializza(root.PACCHETTO, root.SOLUZIONI, null, {persisti: true});
    var lista = el("datalist", {id: "tipi-difetto"});
    L.TIPI_DIFETTO.forEach(function (t) { lista.appendChild(el("option", {value: t})); });
    document.body.appendChild(lista);
    root.addEventListener("hashchange", daHash);
    root.addEventListener("storage", function (e) {
      var mem = memoria();
      if (mem && e.key === chiaveLocale() && sincronizza(mem)) {
        disegna();
      }
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        chiudiLente();
      }
    });
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) {
        fermaCronometro();
        salvaLocale();
      } else {
        avviaCronometro();
      }
    });
    root.addEventListener("beforeunload", function () {
      fermaCronometro();
      salvaLocale();
    });
    var id = decodeURIComponent(root.location.hash.slice(1));
    var inizio = CTX.idx[id] ? id : (STATO.posizione && CTX.idx[STATO.posizione] ? STATO.posizione : null);
    if (inizio) {
      vista.pagina = "schede";
      vista.id = inizio;
      disegna();
      avviaCronometro();
    } else {
      apriPagina("schede");
    }
  }

  root.App = {avvia: avvia, inizializza: inizializza, costruisciVistaScheda: costruisciVistaScheda,
              costruisciRiepilogo: costruisciRiepilogo, importaTesto: importaTesto, salvaOra: salvaLocale,
              contesto: function () { return CTX; }, stato: function () { return STATO; },
              esito: function () { return esitoSalvataggio; }};
})(window);
