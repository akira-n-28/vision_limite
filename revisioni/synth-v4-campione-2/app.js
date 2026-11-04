/* Interfaccia della revisione offline (index.html). Legge window.PACCHETTO, SOLUZIONI ed EVIDENZE (dati/*.js) e usa
   window.Logica (logica.js). Nessuna richiesta di rete: immagini e dati sono file locali della cartella. */
(function (root) {
  "use strict";
  var L = root.Logica;
  var PKG = null, IDX = null, SOL = null, EVI = null, STATO = null;
  var OPZ = {persisti: true};
  var vista = {scheda: "campione", id: null, filtro: "tutti", zoom: "1"};
  var cronometro = {id: null, inizio: 0};
  var voceElenco = {};
  var timerSalvataggio = null;
  var esitoSalvataggio = "";

  var NOMI = {
    corretta: "Correttezza: il target è giusto?",
    coerente_con_annotazioni: "Coerenza con le annotazioni (specifica e layout)",
    rispondibile_da_immagine: "Rispondibilità dall'immagine e dal prompt",
    gravita: "Gravità",
    tipo_difetto: "Tipo di difetto",
    contratto_coppia: "Contratto della coppia"
  };
  var FASI = {
    non_iniziato: ["○", "non iniziato"],
    bozza: ["◔", "primo giudizio in bozza"],
    primo_giudizio: ["◑", "primo giudizio salvato, soluzione nascosta"],
    soluzione_vista: ["◕", "soluzione vista, giudizio finale incompleto"],
    in_corso: ["◑", "in corso"],
    completo: ["●", "completo"]
  };
  var SCHEDE = [["campione", "Campione casuale"], ["mirati", "Casi mirati"], ["decontaminazione", "Decontaminazione"],
                ["riepilogo", "Riepilogo e file"]];

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
    var s = el("select", {disabled: !!disabilitato});
    L.VALORI[nome].forEach(function (x) { s.appendChild(el("option", {value: x[0], testo: x[1]})); });
    s.value = valore;
    s.addEventListener("change", function () { onchange(s.value, s); });
    return s;
  }

  function areaTesto(valore, oninput, disabilitata, righe) {
    var t = el("textarea", {rows: String(righe || 2), disabled: !!disabilitata});
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

  /* ------------------------------------------------------------ immagini ingrandibili */

  function figura(src, didascalia, conLente) {
    var img = el("img", {src: src, alt: didascalia || src, classe: "immagine"});
    var scatola = el("div", {classe: "scatola-immagine"}, img);
    var barra = el("div", {classe: "zoom"});
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
    [["adatta", "adatta"], ["1", "100%"], ["2", "200%"], ["3", "300%"], ["4", "400%"]].forEach(function (l) {
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
    var dimensioni = el("span", {classe: "dimensioni"});
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

  function chiaveLocale() {
    return "revisione:" + PKG.id + ":v" + PKG.versione + ":" + PKG.impronta.slice(0, 16);
  }

  function caricaLocale() {
    if (!OPZ.persisti) {
      return L.nuovoStato(PKG);
    }
    try {
      var testo = root.localStorage.getItem(chiaveLocale());
      if (testo) {
        var r = L.controllaImport(PKG, testo);
        if (!r.errori.length) {
          esitoSalvataggio = "Ripreso il lavoro salvato in questo browser (" + quando(r.stato.aggiornato_il) + ").";
          return r.stato;
        }
        esitoSalvataggio = "Lo stato salvato nel browser non è valido e non è stato caricato: importa il tuo file " +
                           "di avanzamento. (" + r.errori.slice(0, 2).join("; ") + ")";
      }
    } catch (e) {
      esitoSalvataggio = "Salvataggio nel browser non disponibile: esporta spesso il file di avanzamento.";
    }
    return L.nuovoStato(PKG);
  }

  function salvaLocale() {
    if (timerSalvataggio) {
      clearTimeout(timerSalvataggio);
      timerSalvataggio = null;
    }
    STATO.aggiornato_il = L.ora();
    if (!OPZ.persisti) {
      return;
    }
    try {
      root.localStorage.setItem(chiaveLocale(), L.esportaJSON(PKG, STATO));
      esitoSalvataggio = "Salvato nel browser alle " + new Date().toLocaleTimeString("it-IT") + ".";
    } catch (e) {
      esitoSalvataggio = "Salvataggio nel browser non riuscito: scarica subito il file di avanzamento.";
    }
    disegnaBarraSalvataggio();
  }

  function salva() {
    if (timerSalvataggio) {
      clearTimeout(timerSalvataggio);
    }
    timerSalvataggio = setTimeout(salvaLocale, 300);
    STATO.aggiornato_il = L.ora();
  }

  function ultimaEsportazione() {
    try {
      return root.localStorage.getItem(chiaveLocale() + ":esportato") || "";
    } catch (e) {
      return "";
    }
  }

  function segnaEsportazione() {
    try {
      root.localStorage.setItem(chiaveLocale() + ":esportato", STATO.aggiornato_il);
    } catch (e) { /* solo un promemoria */ }
  }

  function scarica(nome, testo, tipo) {
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

  function marcaTempo() {
    return new Date().toISOString().slice(0, 19).replace(/:/g, "-");
  }

  function scaricaAvanzamento() {
    fermaCronometro();
    salvaLocale();
    scarica(PKG.id + "_avanzamento_" + marcaTempo() + ".json", L.esportaJSON(PKG, STATO), "application/json");
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
    barra.appendChild(el("button", {type: "button", classe: "primario", testo: "Scarica il file di avanzamento",
                                    onclick: scaricaAvanzamento}));
  }

  /* ------------------------------------------------------------ tempo per caso */

  function avviaCronometro() {
    cronometro.id = vista.id;
    cronometro.inizio = vista.id && vista.scheda !== "riepilogo" ? Date.now() : 0;
  }

  function fermaCronometro() {
    if (cronometro.id && cronometro.inizio) {
      var ms = Math.min(Date.now() - cronometro.inizio, 30 * 60 * 1000);
      if (ms > 2000) {
        var u = IDX.unita[cronometro.id];
        var d = IDX.decontaminazione[cronometro.id];
        if (u) {
          L.aggiungiTempo(STATO, u, ms, false);
        } else if (d) {
          L.aggiungiTempo(STATO, d.d, ms, true);
        }
        salva();
      }
    }
    cronometro.inizio = 0;
  }

  /* ------------------------------------------------------------ navigazione */

  function elencoScheda(scheda) {
    if (scheda === "decontaminazione") {
      return PKG.decontaminazione;
    }
    return scheda === "riepilogo" ? [] : PKG[scheda];
  }

  function schedaDi(id) {
    if (IDX.unita[id]) {
      return IDX.unita[id].sezione;
    }
    return IDX.decontaminazione[id] ? "decontaminazione" : null;
  }

  function faseDi(id) {
    var u = IDX.unita[id];
    if (u) {
      return L.statoUnita(u, STATO.unita[id]);
    }
    var d = IDX.decontaminazione[id].d;
    return L.controlloDecontam(d, STATO.decontaminazione[id]);
  }

  function vai(id) {
    var scheda = schedaDi(id);
    if (!scheda) {
      return;
    }
    fermaCronometro();
    chiudiLente();
    vista.scheda = scheda;
    vista.id = id;
    STATO.posizione = id;
    if (root.location.hash !== "#" + id) {
      root.location.hash = id;
    }
    disegna();
    root.scrollTo(0, 0);
    avviaCronometro();
  }

  function apriScheda(scheda) {
    fermaCronometro();
    vista.scheda = scheda;
    if (scheda !== "riepilogo") {
      var lista = elencoScheda(scheda);
      if (!vista.id || schedaDi(vista.id) !== scheda) {
        var primo = lista.filter(function (x) { return faseDi(x.id).fase !== "completo"; })[0] || lista[0];
        vista.id = primo.id;
      }
      if (root.location.hash !== "#" + vista.id) {
        root.location.hash = vista.id;
      }
    }
    disegna();
    avviaCronometro();
  }

  function sposta(delta) {
    var lista = elencoScheda(vista.scheda);
    var i = lista.map(function (x) { return x.id; }).indexOf(vista.id);
    var j = i + delta;
    if (j >= 0 && j < lista.length) {
      vai(lista[j].id);
    }
  }

  function prossimoDaCompletare() {
    var lista = elencoScheda(vista.scheda);
    var i = lista.map(function (x) { return x.id; }).indexOf(vista.id);
    for (var n = 1; n <= lista.length; n++) {
      var x = lista[(i + n) % lista.length];
      if (faseDi(x.id).fase !== "completo") {
        vai(x.id);
        return true;
      }
    }
    return false;
  }

  function daHash() {
    var id = decodeURIComponent(root.location.hash.slice(1));
    if (id && id !== vista.id && schedaDi(id)) {
      vai(id);
    }
  }

  /* ------------------------------------------------------------ schede ed elenco */

  function disegnaSchede() {
    var nav = $("schede");
    if (!nav) {
      return;
    }
    var r = L.riepilogo(PKG, STATO);
    nav.innerHTML = "";
    SCHEDE.forEach(function (s) {
      var testo = s[1];
      var c = r[s[0]];
      if (c) {
        testo += " " + (c.completo || 0) + "/" + c.totale;
        if (c.incerto) {
          testo += " · " + c.incerto + " incerti";
        }
      }
      var b = el("button", {type: "button", classe: "scheda" + (vista.scheda === s[0] ? " attiva" : ""), testo: testo,
                            onclick: function () { apriScheda(s[0]); }});
      if (c) {
        var barra = el("span", {classe: "avanzamento"});
        barra.appendChild(el("span", {classe: "pieno", style: "width:" + (100 * (c.completo || 0) / c.totale) + "%"}));
        b.appendChild(barra);
      }
      nav.appendChild(b);
    });
  }

  function segno(id) {
    var s = faseDi(id);
    var f = FASI[s.fase] || ["?", s.fase];
    return {testo: f[0] + (s.incerto ? "?" : "") + (s.non_giudicabile ? "!" : ""), titolo: f[1] +
            (s.incerto ? ", con valori incerti" : "") + (s.non_giudicabile ? ", con valori non giudicabili" : ""),
            fase: s.fase, incerto: s.incerto || s.non_giudicabile};
  }

  function passaFiltro(id) {
    var s = segno(id);
    return vista.filtro === "tutti" || (vista.filtro === "da_completare" && s.fase !== "completo")
      || (vista.filtro === "incerti" && s.incerto) || (vista.filtro === "completi" && s.fase === "completo");
  }

  function disegnaElenco() {
    var aside = $("elenco");
    if (!aside) {
      return;
    }
    aside.innerHTML = "";
    voceElenco = {};
    if (vista.scheda === "riepilogo") {
      aside.hidden = true;
      return;
    }
    aside.hidden = false;
    var filtro = el("select", {});
    [["tutti", "tutti"], ["da_completare", "da completare"], ["incerti", "incerti o non giudicabili"],
     ["completi", "completi"]].forEach(function (x) { filtro.appendChild(el("option", {value: x[0], testo: x[1]})); });
    filtro.value = vista.filtro;
    filtro.addEventListener("change", function () {
      vista.filtro = filtro.value;
      disegnaElenco();
    });
    aside.appendChild(el("div", {classe: "filtro"}, "Mostra ", filtro));
    aside.appendChild(el("p", {classe: "legenda", testo: "○ non iniziato · ◔ bozza · ◑ primo giudizio salvato · " +
                                                     "◕ soluzione vista · ● completo · ? incerto · ! non giudicabile"}));
    var gruppo = null, lista = null;
    elencoScheda(vista.scheda).forEach(function (x) {
      var g = x.gruppo || "candidati";
      if (g !== gruppo) {
        gruppo = g;
        aside.appendChild(el("h4", {testo: g}));
        lista = el("ul", {});
        aside.appendChild(lista);
      }
      if (!passaFiltro(x.id) && x.id !== vista.id) {
        return;
      }
      var s = segno(x.id);
      var b = el("button", {type: "button", classe: "voce" + (x.id === vista.id ? " corrente" : ""), title: s.titolo,
                            onclick: function () { vai(x.id); }},
                 el("span", {classe: "segno", testo: s.testo}), " " + x.id);
      voceElenco[x.id] = b;
      lista.appendChild(el("li", {}, b));
    });
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
    disegnaSchede();
  }

  function navigazione(id) {
    var lista = elencoScheda(vista.scheda);
    var i = lista.map(function (x) { return x.id; }).indexOf(id);
    var msg = el("span", {classe: "messaggio"});
    return el("div", {classe: "navigazione"},
              el("button", {type: "button", disabled: i <= 0, testo: "← precedente", onclick: function () { sposta(-1); }}),
              el("span", {classe: "posizione", testo: (i + 1) + " di " + lista.length}),
              el("button", {type: "button", disabled: i >= lista.length - 1, testo: "successivo →",
                            onclick: function () { sposta(1); }}),
              el("button", {type: "button", testo: "prossimo da completare", onclick: function () {
                if (!prossimoDaCompletare()) {
                  messaggio(msg, ["Tutti i casi di questa scheda sono completi."], "ok");
                }
              }}), msg);
  }

  /* ------------------------------------------------------------ unità: primo passaggio */

  function bloccoPrompt(u) {
    if (!u.prompt) {
      return el("div", {classe: "blocco-prompt"}, el("h3", {testo: "Prompt"}),
                el("p", {classe: "nota", testo: "Nessun prompt: caso sulla scena intera di un record che il loader " +
                                               "di training non legge."}));
    }
    var pre = el("pre", {classe: "prompt"});
    var righe = u.prompt.split("\n");
    righe.forEach(function (r, i) {
      var evidenzia = u.domande_nel_prompt.length > 1 && u.numero_nel_prompt && r.indexOf(u.numero_nel_prompt + ". ") === 0;
      pre.appendChild(el("span", {classe: evidenzia ? "riga evidenziata" : "riga", testo: r}));
      if (i < righe.length - 1) {
        pre.appendChild(document.createTextNode("\n"));
      }
    });
    var info = [];
    if (u.numero_nel_prompt && u.domande_nel_prompt.length > 1) {
      info.push("QA campionata: domanda " + u.numero_nel_prompt + " di " + u.domande_nel_prompt.length +
                " (evidenziata)");
    }
    info.push("origine del prompt: " + u.origine_prompt);
    info.push(u.consegnato_dal_loader ? "il loader di training consegna questa scena" :
              "il loader di training non consegna questa QA o questa scena");
    return el("div", {classe: "blocco-prompt"},
              el("h3", {testo: "Prompt esatto (testo dell'utente, dopo l'immagine)"}), pre,
              el("p", {classe: "nota", testo: info.join(" · ")}));
  }

  function metaUnita(u) {
    var righe = [["unità", u.id], [u.sezione === "campione" ? "strato" : "categoria", u.gruppo], ["tipo", u.tipo],
                 ["famiglia", u.famiglia], ["livello", u.livello === "qa" ? "una QA" : "scena intera"]];
    if (u.posizione_qa !== null) {
      righe.push(["posizione della QA nel record", String(u.posizione_qa)]);
    }
    if (u.sezione === "campione") {
      righe.push(["rango", u.rango.slice(0, 16) + "…"], ["inclusione", u.p_inclusione]);
    }
    u.immagini.forEach(function (im) {
      righe.push([im.etichetta || "record", im.record + " · " + im.insieme + "/" + im.split + " · " + im.stato]);
    });
    return el("table", {classe: "meta"}, righe.map(function (r) {
      return el("tr", {}, el("th", {testo: r[0]}), el("td", {testo: r[1]}));
    }));
  }

  function sezionePrima(u) {
    var v = L.voce(STATO, u, false);
    var bloccato = !!(v && v.soluzione_mostrata_il);
    var box = el("section", {classe: "passaggio prima"},
                 el("h3", {testo: "1. Primo giudizio, senza la soluzione"}));
    var msg = el("p", {classe: "messaggio"});
    var rivela = el("button", {type: "button", classe: "rivela", testo: "Mostra soluzione e materiale di confronto"});
    var motivo = el("span", {classe: "aiuto"});
    function aggiornaRivela() {
      var p = L.puoMostrare(STATO, u);
      rivela.disabled = !p.ok;
      motivo.textContent = p.ok ? "Il primo giudizio resterà fissato." : p.motivo;
    }
    function modifica(k, nome, valore) {
      var errori = L.aggiornaBozza(STATO, u, k, nome, valore);
      if (errori.length) {
        messaggio(msg, errori, "errore");
      }
      salva();
      aggiornaRivela();
      aggiornaSegno(u.id);
    }
    u.varianti.forEach(function (k) {
      var b = bloccato ? v.prima.varianti[k] : (v ? v.bozza[k] : {risposta: "", non_riesco: false, leggibilita: L.NON,
                                                                  note: ""});
      var fs = el("fieldset", {classe: "variante"},
                  el("legend", {testo: k ? "Variante " + k + " · " + u.record_per_variante[k] : "Unità " + u.id}));
      var spunta = el("input", {type: "checkbox", disabled: bloccato});
      spunta.checked = b.non_riesco;
      spunta.addEventListener("change", function () { modifica(k, "non_riesco", spunta.checked); });
      fs.appendChild(campo(u.livello === "qa" ? "Risposta" : "Risposta o descrizione (facoltativa)", L.AIUTI.risposta,
                           areaTesto(b.risposta, function (t) { modifica(k, "risposta", t); }, bloccato)));
      fs.appendChild(el("label", {classe: "campo spunta"}, spunta, el("span", {testo: " non riesco a rispondere"}),
                        el("span", {classe: "aiuto", testo: L.AIUTI.non_riesco})));
      fs.appendChild(campo("Leggibilità e rispondibilità (giudizio iniziale)", L.AIUTI.leggibilita,
                           scelta("leggibilita", b.leggibilita, function (x) { modifica(k, "leggibilita", x); },
                                  bloccato)));
      fs.appendChild(campo("Note del primo giudizio", L.AIUTI.note_prima,
                           areaTesto(b.note, function (t) { modifica(k, "note", t); }, bloccato)));
      box.appendChild(fs);
    });
    if (bloccato) {
      box.appendChild(el("p", {classe: "nota", testo: "Primo giudizio salvato il " + quando(v.prima.salvata_il) +
        "; soluzione mostrata il " + quando(v.soluzione_mostrata_il) + ". Il primo giudizio non si modifica più: i " +
        "ripensamenti vanno nel giudizio finale, anche nel campo «risposta dopo la soluzione»."}));
      return box;
    }
    box.appendChild(el("div", {classe: "azioni"},
                       el("button", {type: "button", classe: "primario", testo: "Salva il primo giudizio",
                                     onclick: function () {
                                       var errori = L.salvaPrima(STATO, u);
                                       salvaLocale();
                                       if (errori.length) {
                                         messaggio(msg, errori, "errore");
                                       } else {
                                         messaggio(msg, ["Primo giudizio salvato. Ora puoi mostrare la soluzione."],
                                                   "ok");
                                       }
                                       aggiornaRivela();
                                       aggiornaSegno(u.id);
                                     }})));
    box.appendChild(msg);
    rivela.addEventListener("click", function () {
      if (!root.confirm("Dopo aver visto la soluzione il primo giudizio resta fissato e non si modifica più. " +
                        "Mostrare la soluzione?")) {
        return;
      }
      var errori = L.mostraSoluzione(STATO, u);
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

  /* ------------------------------------------------------------ unità: soluzione e giudizio finale */

  function dettaglio(titolo, oggetto) {
    return el("details", {classe: "json"}, el("summary", {testo: titolo}),
              el("pre", {testo: JSON.stringify(oggetto, null, 1)}));
  }

  function materialeRecord(im) {
    var e = EVI[im.record];
    var det = el("details", {classe: "materiale", open: true},
                 el("summary", {testo: "Materiale del record " + im.record + (im.etichetta ? " (" + im.etichetta + ")" :
                                                                              "")}));
    if (!e) {
      det.appendChild(el("p", {classe: "errore", testo: "Materiale non disponibile per questo record: dichiara " +
                                                       "«non verificabile» dove serve."}));
      return det;
    }
    var meta;
    try {
      meta = JSON.parse(e.json);
    } catch (err) {
      det.appendChild(el("p", {classe: "errore", testo: "JSON del record non leggibile."}));
      return det;
    }
    var qa = (meta.qa || []).slice();
    if (meta.pair) {
      qa.push(meta.pair);
    }
    var esiti = e.checker && e.checker.questions ? e.checker.questions : [];
    var t = el("table", {classe: "qa"}, el("tr", {}, ["pos.", "tipo", "domanda", "opzioni", "risposta attesa",
                                                       "esito del checker"].map(function (h) {
      return el("th", {testo: h}); })));
    qa.forEach(function (q, i) {
      t.appendChild(el("tr", {classe: i === im.posizione ? "evidenziata" : ""},
                       [String(i), q.type, q.q, (q.options || []).join(" / "), el("strong", {testo: q.answer}),
                        esiti[i] ? esiti[i][2] : ""].map(function (c) { return el("td", {}, c); })));
    });
    det.appendChild(el("h4", {testo: "Domande del record con le risposte attese (JSON dello shard)"}));
    det.appendChild(t);
    det.appendChild(dettaglio("Specifica della scena (spec)", meta.spec));
    det.appendChild(dettaglio("Layout disegnato (oggetti, riquadri in pixel)", meta.layout));
    det.appendChild(dettaglio("Riga del checker (candidato v4)", e.checker));
    det.appendChild(dettaglio("JSON completo del record (sha256 " + e.json_sha256.slice(0, 16) + "…)", meta));
    return det;
  }

  function campoDifetto(valore, oninput) {
    var i = el("input", {type: "text", list: "tipi-difetto"});
    i.value = valore;
    i.addEventListener("input", function () { oninput(i.value); });
    return i;
  }

  function moduloDopo(u, k, aggiornaStato) {
    var v = L.voce(STATO, u, false);
    var d = v.dopo.varianti[k];
    var p = v.prima.varianti[k];
    var nome = k ? "variante " + k : "unità";
    function cambia(c, valore) {
      var errori = L.aggiornaDopo(STATO, u, k, c, valore);
      salva();
      aggiornaStato(errori);
    }
    var fs = el("fieldset", {classe: "variante"}, el("legend", {testo: "Giudizio finale · " + nome}));
    fs.appendChild(el("p", {classe: "ricordo"}, "Risposta iniziale (indipendente): ",
                      el("strong", {testo: p.non_riesco ? "non riesco a rispondere" : (p.risposta || "—")}),
                      " · giudizio iniziale: " + etichetta("leggibilita", p.leggibilita)));
    ["corretta", "coerente_con_annotazioni", "rispondibile_da_immagine"].forEach(function (c) {
      fs.appendChild(campo(NOMI[c], L.AIUTI[c], scelta("giudizio", d[c], function (x) { cambia(c, x); })));
    });
    fs.appendChild(campo(NOMI.gravita, L.AIUTI.gravita, scelta("gravita", d.gravita, function (x) { cambia("gravita", x); })));
    fs.appendChild(campo(NOMI.tipo_difetto, L.AIUTI.tipo_difetto,
                         campoDifetto(d.tipo_difetto, function (x) { cambia("tipo_difetto", x); })));
    fs.appendChild(campo("Note del giudizio finale", L.AIUTI.note_dopo,
                         areaTesto(d.note, function (x) { cambia("note", x); }, false, 3)));
    fs.appendChild(campo("Risposta dopo la soluzione (non indipendente)", L.AIUTI.risposta_dopo_soluzione,
                         areaTesto(d.risposta_dopo_soluzione, function (x) { cambia("risposta_dopo_soluzione", x); })));
    return fs;
  }

  function moduloCoppia(u, aggiornaStato) {
    var v = L.voce(STATO, u, false);
    var c = v.dopo.coppia;
    function cambia(nome, valore) {
      var errori = L.aggiornaDopo(STATO, u, "coppia", nome, valore);
      salva();
      aggiornaStato(errori);
    }
    var fs = el("fieldset", {classe: "variante coppia"}, el("legend", {testo: "Giudizio sulla coppia (a e b insieme)"}));
    fs.appendChild(campo(NOMI.contratto_coppia, L.AIUTI.contratto_coppia,
                         scelta("giudizio", c.contratto_coppia, function (x) { cambia("contratto_coppia", x); })));
    fs.appendChild(campo(NOMI.gravita, L.AIUTI.gravita, scelta("gravita", c.gravita, function (x) { cambia("gravita", x); })));
    fs.appendChild(campo(NOMI.tipo_difetto, L.AIUTI.tipo_difetto,
                         campoDifetto(c.tipo_difetto, function (x) { cambia("tipo_difetto", x); })));
    fs.appendChild(campo("Note sulla coppia", L.AIUTI.note_dopo, areaTesto(c.note, function (x) { cambia("note", x); },
                                                                         false, 3)));
    return fs;
  }

  function sezioneDopo(u) {
    var v = L.voce(STATO, u, false);
    if (!v || !v.soluzione_mostrata_il) {
      return el("section", {classe: "passaggio dopo nascosto"},
                el("h3", {testo: "2. Soluzione, materiale di confronto e giudizio finale"}),
                el("p", {classe: "nota", testo: "Si apre dopo aver salvato il primo giudizio e premuto «Mostra " +
                  "soluzione». Fino ad allora restano nascosti target, opzioni ausiliarie, esito del checker, motivi " +
                  "e misure del caso."}));
    }
    var s = SOL[u.id];
    var box = el("section", {classe: "passaggio dopo soluzione"},
                 el("h3", {testo: "2. Soluzione, materiale di confronto e giudizio finale"}));
    if (s.motivo || s.misura) {
      box.appendChild(el("p", {classe: "motivo"}, el("strong", {testo: "Perché è un caso mirato: "}),
                         (s.motivo || "") + (s.misura ? " (misura " + s.misura + ")" : "")));
    }
    if (s.varianti.length) {
      var t = el("table", {classe: "soluzioni"}, el("tr", {}, ["variante", "record", "risposta attesa", "opzioni",
                                                                "opzioni nel testo della domanda", "esito del checker",
                                                                "stato del record"].map(function (h) {
        return el("th", {testo: h}); })));
      s.varianti.forEach(function (x) {
        t.appendChild(el("tr", {}, [x.variante || "—", x.record, el("strong", {testo: x.risposta}),
                                    x.opzioni.join(" / "), x.opzioni_nel_testo_della_domanda ? "sì" : "no",
                                    x.esito_checker, x.stato].map(function (c) { return el("td", {}, c); })));
      });
      box.appendChild(t);
    } else {
      box.appendChild(el("p", {classe: "nota", testo: "Caso sulla scena intera: non c'è una sola risposta attesa. Le " +
                                                     "domande della scena con le risposte sono nel materiale qui sotto."}));
    }
    u.immagini.forEach(function (im) {
      box.appendChild(materialeRecord({record: im.record, etichetta: im.etichetta, posizione: u.posizione_qa}));
    });
    var stato = el("div", {classe: "stato-caso"});
    function aggiornaStato(errori) {
      var st = L.statoUnita(u, L.voce(STATO, u, false));
      stato.innerHTML = "";
      stato.appendChild(el("strong", {testo: FASI[st.fase][1]}));
      if (st.mancanti.length) {
        stato.appendChild(el("span", {testo: " · mancano: " + st.mancanti.join(", ")}));
      }
      if (errori && errori.length) {
        stato.appendChild(el("p", {classe: "errore", testo: errori.join(" · ")}));
      }
      aggiornaSegno(u.id);
    }
    u.varianti.forEach(function (k) { box.appendChild(moduloDopo(u, k, aggiornaStato)); });
    if (u.coppia) {
      box.appendChild(moduloCoppia(u, aggiornaStato));
    }
    box.appendChild(stato);
    aggiornaStato([]);
    return box;
  }

  function costruisciVistaUnita(u) {
    var nodo = el("div", {classe: "caso", "data-id": u.id});
    nodo.appendChild(el("h2", {testo: u.id + (u.coppia ? " · coppia" : "")}));
    nodo.appendChild(navigazione(u.id));
    nodo.appendChild(el("p", {classe: "consegna", testo: u.consegna}));
    nodo.appendChild(metaUnita(u));
    nodo.appendChild(el("div", {classe: u.immagini.length > 1 ? "galleria doppia" : "galleria"},
                        u.immagini.map(function (im) {
                          return figura(im.immagine, (im.etichetta ? im.etichetta + " · " : "") + im.record);
                        })));
    nodo.appendChild(bloccoPrompt(u));
    nodo.appendChild(sezionePrima(u));
    nodo.appendChild(sezioneDopo(u));
    return nodo;
  }

  /* ------------------------------------------------------------ decontaminazione */

  function costruisciVistaDecontam(d) {
    var nodo = el("div", {classe: "caso decontaminazione", "data-id": d.id});
    nodo.appendChild(el("h2", {testo: d.id + " · " + d.record}));
    nodo.appendChild(navigazione(d.id));
    nodo.appendChild(el("p", {classe: "consegna", testo: "Il record sintetico ha lo stesso contenuto di un'immagine " +
      "del pannello? Esamina ogni corrispondenza. Nessun esito di questa interfaccia cambia ammissioni, quarantene o " +
      "gruppi: gli esiti entrano solo in una versione nuova del manifest, dopo la loro verifica."}));
    var info = [["record", d.record], ["record_id", d.record_id], ["insieme / split / stato", d.insieme + " / " +
                 d.split + " / " + d.stato], ["famiglia", d.famiglia], ["esito della scansione", d.esito +
                 " (gruppo: " + d.esito_gruppo + ")"], ["corrispondenze", JSON.stringify(d.corrispondenze) + " · " +
                 d.elenco.length + " nel foglio, " + d.corrispondenze_nella_scansione + " nella scansione"],
                ["ruoli nel pannello", d.ruoli], ["materiale", d.materiale]];
    nodo.appendChild(el("table", {classe: "meta"}, info.map(function (r) {
      return el("tr", {}, el("th", {testo: r[0]}), el("td", {testo: String(r[1])}));
    })));
    nodo.appendChild(el("p", {classe: "orientamento"}, el("strong", {testo: "Solo per orientarsi, non è un giudizio: "}),
                        "somiglianza dei pixel a 64×64 " + (d.somiglianza_pixel === null ? "—" : d.somiglianza_pixel) +
                        ", rapporto d'aspetto " + (d.rapporto_aspetto === null ? "—" : d.rapporto_aspetto) +
                        ", elemento più simile " + (d.elemento_piu_simile || "—") + "."));
    var mancante = L.materialeMancante(d);
    if (mancante.length) {
      nodo.appendChild(el("p", {classe: "errore", testo: "Materiale mancante o incompleto: " + mancante.join("; ") +
                                                       ". Il «no» non è ammesso."}));
    }
    var modulo = el("div", {classe: "esito-decontaminazione"});
    var schede = el("div", {classe: "corrispondenze"});
    function disegnaModulo(avvisi) {
      var v = L.voceDecontam(STATO, d, false);
      var esito = v ? v.esito : L.NON;
      var s = L.controlloDecontam(d, v);
      modulo.innerHTML = "";
      modulo.appendChild(el("h3", {testo: "Esito del record"}));
      modulo.appendChild(el("p", {classe: "nota", testo: "Esaminate " + s.esaminate + " corrispondenze su " +
                                                        s.totale + "."}));
      var msg = el("p", {classe: "messaggio"});
      var sel = scelta("esito_decontaminazione", esito, function (x) {
        var errori = L.impostaEsitoDecontam(STATO, d, "esito", x);
        salva();
        if (errori.length) {
          sel.value = esito;
          messaggio(msg, errori, "errore");
        } else {
          disegnaModulo([]);
        }
        aggiornaSegno(d.id);
      });
      modulo.appendChild(campo("Stesso contenuto di un elemento del pannello?", L.AIUTI.esito_decontaminazione, sel));
      if (esito === "si") {
        var dec = el("select", {});
        dec.appendChild(el("option", {value: "", testo: "— scegli la corrispondenza decisiva —"}));
        d.elenco.forEach(function (x) {
          var stesso = v.corrispondenze[x.id].esame === "stesso_contenuto";
          dec.appendChild(el("option", {value: x.id, disabled: !stesso,
                                        testo: x.id + " · " + x.elemento + (stesso ? "" : " (non esaminata come stesso " +
                                                                              "contenuto)")}));
        });
        dec.value = v.decisiva;
        dec.addEventListener("change", function () {
          L.impostaEsitoDecontam(STATO, d, "decisiva", dec.value);
          salva();
          disegnaModulo([]);
          aggiornaSegno(d.id);
        });
        modulo.appendChild(campo("Corrispondenza decisiva", L.AIUTI.decisiva, dec));
      }
      modulo.appendChild(campo("Note sul record", "Più righe ammesse.", areaTesto(v ? v.note : "", function (x) {
        L.impostaEsitoDecontam(STATO, d, "note", x);
        salva();
        aggiornaSegno(d.id);
      }, false, 3)));
      modulo.appendChild(el("p", {classe: "stato-caso"}, el("strong", {testo: FASI[s.fase][1]}),
                            s.errori.length ? " · " + s.errori.join(" · ") : ""));
      if (avvisi && avvisi.length) {
        messaggio(msg, avvisi, "errore");
      }
      modulo.appendChild(msg);
    }
    d.elenco.forEach(function (x) {
      var v = L.voceDecontam(STATO, d, false);
      var c = v ? v.corrispondenze[x.id] : {esame: "non_esaminata", note: ""};
      var scheda = el("div", {classe: "corrispondenza" + (x.nell_anteprima_del_manifest ? "" : " fuori-anteprima"),
                              id: "c-" + x.id});
      scheda.appendChild(el("h4", {testo: x.id + " · " + x.n + " di " + x.di}));
      scheda.appendChild(el("table", {classe: "meta"}, [
        ["tipo e distanza", x.tipo + (x.distanza === null ? "" : " d=" + x.distanza)], ["benchmark", x.benchmark],
        ["elemento del pannello", x.elemento], ["posizione dell'immagine", String(x.posizione_immagine)],
        ["voce dell'indice", x.voce_indice], ["ruolo", x.ruolo + (x.metrica_primaria === null ? "" :
                                                                    " · metrica primaria " + x.metrica_primaria)],
        ["nell'anteprima del manifest", x.nell_anteprima_del_manifest ? "sì" : "no (solo nella scansione completa)"],
        ["problemi", x.problemi || "nessuno"]].map(function (r) {
          return el("tr", {}, el("th", {testo: r[0]}), el("td", {testo: r[1]}));
        })));
      if (x.immagine) {
        scheda.appendChild(figura(x.immagine, x.id + " · " + x.elemento));
      } else {
        scheda.appendChild(el("p", {classe: "errore", testo: "Immagine del pannello assente: segna «non giudicabile»."}));
      }
      var stato = el("span", {classe: "aiuto"});
      var sel = scelta("esame", c.esame, function (val) {
        var r = L.aggiornaEsame(STATO, d, x.id, "esame", val);
        salva();
        scheda.classList.toggle("esaminata", val !== "non_esaminata");
        disegnaModulo(r.errori.concat(r.avvisi));
        aggiornaSegno(d.id);
      });
      scheda.classList.toggle("esaminata", c.esame !== "non_esaminata");
      scheda.appendChild(campo("Esame di questa corrispondenza", L.AIUTI.esame, sel));
      scheda.appendChild(campo("Note", "", areaTesto(c.note, function (val) {
        L.aggiornaEsame(STATO, d, x.id, "note", val);
        salva();
        aggiornaSegno(d.id);
      })));
      scheda.appendChild(stato);
      schede.appendChild(scheda);
    });
    var colonne = el("div", {classe: "colonne"},
                     el("div", {classe: "colonna-record"}, el("h3", {testo: "Record sintetico"}),
                        figura(d.immagine, d.record)),
                     el("div", {classe: "colonna-pannello"},
                        el("h3", {testo: "Immagini candidate del pannello (" + d.elenco.length + ")"}), schede));
    nodo.appendChild(colonne);
    disegnaModulo([]);
    nodo.appendChild(modulo);
    return nodo;
  }

  /* ------------------------------------------------------------ riepilogo, esportazione, importazione */

  function tabellaRiepilogo(r) {
    var colonne = [["totale", "totale"], ["completo", "completi"], ["in_corso", "in corso"],
                   ["non_iniziato", "non iniziati"], ["incerto", "incerti"], ["non_giudicabile", "non giudicabili"],
                   ["non_verificabile", "non verificabili"]];
    function inCorso(c) {
      return (c.bozza || 0) + (c.primo_giudizio || 0) + (c.soluzione_vista || 0) + (c.in_corso || 0);
    }
    var t = el("table", {classe: "riepilogo"}, el("tr", {}, el("th", {testo: "sezione / gruppo"}),
                                                   colonne.map(function (c) { return el("th", {testo: c[1]}); })));
    function riga(nome, c, classe) {
      t.appendChild(el("tr", {classe: classe || ""}, el("td", {testo: nome}), colonne.map(function (col) {
        return el("td", {testo: String(col[0] === "in_corso" ? inCorso(c) : (c[col[0]] || 0))});
      })));
    }
    riga("Campione casuale", r.campione, "totale");
    Object.keys(r.campione.gruppi).forEach(function (g) { riga("  " + g, r.campione.gruppi[g]); });
    riga("Casi mirati", r.mirati, "totale");
    Object.keys(r.mirati.gruppi).forEach(function (g) { riga("  " + g, r.mirati.gruppi[g]); });
    riga("Decontaminazione", r.decontaminazione, "totale");
    return t;
  }

  function costruisciRiepilogo() {
    var r = L.riepilogo(PKG, STATO);
    var nodo = el("div", {classe: "caso riepilogo"});
    nodo.appendChild(el("h2", {testo: "Riepilogo, salvataggio e restituzione"}));
    nodo.appendChild(tabellaRiepilogo(r));
    nodo.appendChild(el("p", {classe: "nota", testo: "Corrispondenze della decontaminazione esaminate: " +
                                                     (r.decontaminazione.corrispondenze_esaminate || 0) + " su " +
                                                     r.decontaminazione.corrispondenze + "."}));
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
      el("button", {type: "button", testo: "Scarica esiti.tsv", onclick: function () {
        scarica("esiti_" + marcaTempo() + ".tsv", L.tsvEsiti(PKG, STATO), "text/tab-separated-values");
      }}),
      el("button", {type: "button", testo: "Scarica esiti_decontaminazione.tsv", onclick: function () {
        scarica("esiti_decontaminazione_" + marcaTempo() + ".tsv", L.tsvDecontaminazione(PKG, STATO),
                "text/tab-separated-values");
      }})));
    nodo.appendChild(el("p", {classe: "nota", testo: "Il file JSON contiene tutto il lavoro, compresi primo giudizio, " +
      "risposte dopo la soluzione, esami delle corrispondenze, tempi ed eventi: è la copia da conservare e da " +
      "restituire. I TSV hanno le colonne dei moduli originali; nelle celle gli a capo sono \\n, le tabulazioni \\t " +
      "e la barra rovesciata \\\\. Un campo non compilato vale «non_compilato», mai approvazione."}));
    var copia = el("textarea", {rows: "6", readonly: true, classe: "copia"});
    nodo.appendChild(el("details", {}, el("summary", {testo: "Se lo scaricamento non funziona: copia il JSON a mano"}),
      el("p", {classe: "nota", testo: "Premi il pulsante, seleziona tutto il testo e incollalo in un file di testo " +
                                      "con estensione .json."}),
      el("button", {type: "button", testo: "Mostra il JSON", onclick: function () {
        copia.value = L.esportaJSON(PKG, STATO);
        copia.select();
      }}), copia));
    nodo.appendChild(el("h3", {testo: "Riprendere da un file di avanzamento"}));
    var input = el("input", {type: "file", accept: ".json,application/json"});
    var conflitti = el("div", {});
    input.addEventListener("change", function () {
      var f = input.files && input.files[0];
      if (!f) {
        return;
      }
      var lettore = new FileReader();
      lettore.onload = function () { importa(String(lettore.result), msg, conflitti); };
      lettore.onerror = function () { messaggio(msg, ["File non leggibile."], "errore"); };
      lettore.readAsText(f, "utf-8");
      input.value = "";
    });
    nodo.appendChild(campo("Importa il file JSON scaricato in una sessione precedente", "Si controllano pacchetto, " +
      "versione, impronta e identificativi; le risposte in conflitto non si sovrascrivono mai in silenzio.", input));
    nodo.appendChild(msg);
    nodo.appendChild(conflitti);
    nodo.appendChild(el("h3", {testo: "Che cosa ricaricare su GitHub"}));
    nodo.appendChild(el("p", {testo: "Nella cartella risultati/ di questo pacchetto: l'ultimo file di avanzamento " +
      "(…_avanzamento_….json) e, se vuoi, i due TSV scaricati per ultimi. Non modificare gli altri file del " +
      "pacchetto. Le istruzioni complete sono in LEGGIMI.md."}));
    nodo.appendChild(el("p", {classe: "nota", testo: "Pacchetto " + PKG.id + " v" + PKG.versione + " · impronta " +
                                                     PKG.impronta + " · campione " + PKG.sorgente.campione +
                                                     " (report.json sha256 " + PKG.sorgente.report_sha256 + ")"}));
    return nodo;
  }

  function importa(testo, msg, box) {
    box.innerHTML = "";
    var r = L.controllaImport(PKG, testo);
    if (r.errori.length) {
      messaggio(msg, ["File rifiutato: " + r.errori.slice(0, 8).join("; ") +
                      (r.errori.length > 8 ? " e altri " + (r.errori.length - 8) : "")], "errore");
      return;
    }
    var c = L.confronta(STATO, r.stato);
    if (!c.conflitti.length) {
      STATO = c.unito;
      salvaLocale();
      messaggio(msg, ["File importato: " + c.nuove + " casi aggiunti, " + c.uguali + " già uguali, " +
                      c.solo_correnti + " presenti solo nel browser e conservati."], "ok");
      disegna();
      return;
    }
    messaggio(msg, [c.conflitti.length + " casi hanno contenuti diversi nel browser e nel file: niente è stato " +
                    "importato."], "errore");
    box.appendChild(el("ul", {}, c.conflitti.slice(0, 50).map(function (x) {
      return el("li", {testo: x.id + ": nel browser modificato il " + quando(x.corrente) + ", nel file il " +
                              quando(x.file)});
    })));
    box.appendChild(el("div", {classe: "azioni"},
      el("button", {type: "button", testo: "Mantieni lo stato del browser (ignora il file)", onclick: function () {
        box.innerHTML = "";
        messaggio(msg, ["Importazione annullata."], "");
      }}),
      el("button", {type: "button", testo: "Sostituisci tutto con il file (prima scarica una copia del browser)",
                    onclick: function () {
                      scarica(PKG.id + "_copia_del_browser_" + marcaTempo() + ".json", L.esportaJSON(PKG, STATO),
                              "application/json");
                      STATO = r.stato;
                      salvaLocale();
                      box.innerHTML = "";
                      messaggio(msg, ["Stato sostituito con il file; la copia precedente è stata scaricata."], "ok");
                      disegna();
                    }})));
  }

  /* ------------------------------------------------------------ disegno */

  function disegna() {
    disegnaSchede();
    disegnaElenco();
    disegnaBarraSalvataggio();
    var main = $("caso");
    if (!main) {
      return;
    }
    main.innerHTML = "";
    if (vista.scheda === "riepilogo") {
      main.appendChild(costruisciRiepilogo());
      return;
    }
    var u = IDX.unita[vista.id];
    if (u) {
      main.appendChild(costruisciVistaUnita(u));
    } else if (IDX.decontaminazione[vista.id]) {
      main.appendChild(costruisciVistaDecontam(IDX.decontaminazione[vista.id].d));
    }
  }

  function inizializza(pkg, sol, evi, stato, opzioni) {
    PKG = pkg;
    SOL = sol;
    EVI = evi;
    IDX = L.indice(pkg);
    OPZ = opzioni || {persisti: true};
    STATO = stato || caricaLocale();
  }

  function avvia() {
    if (!root.PACCHETTO || !root.SOLUZIONI || !root.EVIDENZE || !root.Logica) {
      document.body.appendChild(el("p", {classe: "errore", testo: "Dati del pacchetto non caricati. Estrai tutta la " +
        "cartella (con dati/ e immagini/) e apri index.html da lì."}));
      return;
    }
    inizializza(root.PACCHETTO, root.SOLUZIONI, root.EVIDENZE, null, {persisti: true});
    var lista = el("datalist", {id: "tipi-difetto"});
    L.TIPI_DIFETTO.forEach(function (t) { lista.appendChild(el("option", {value: t})); });
    document.body.appendChild(lista);
    root.addEventListener("hashchange", daHash);
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
    var inizio = schedaDi(id) ? id : (STATO.posizione && schedaDi(STATO.posizione) ? STATO.posizione : null);
    if (inizio) {
      vista.scheda = schedaDi(inizio);
      vista.id = inizio;
      disegna();
      avviaCronometro();
    } else {
      apriScheda("campione");
    }
  }

  root.App = {avvia: avvia, inizializza: inizializza, costruisciVistaUnita: costruisciVistaUnita,
              costruisciVistaDecontam: costruisciVistaDecontam, costruisciRiepilogo: costruisciRiepilogo,
              stato: function () { return STATO; }};
})(window);
