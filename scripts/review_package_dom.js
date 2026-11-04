/* DOM simulato minimo per eseguire app.js e test.js dei pacchetti di revisione (revisioni/*) con Duktape, fuori da un
   browser (scripts/review_package.py esegui-js). Copre solo ciò che usano app.js e test.js: elementi con attributi,
   classi, figli, testo, value/checked/disabled/hidden, eventi, querySelector(All) con selettori semplici (tag, .classe,
   #id, [attributo], [attributo=valore], discendenti, liste separate da virgole), timer manuali. Non disegna, non carica
   immagini e non fa richieste: non è una prova in un browser. */
var window = this;
(function (root) {
  "use strict";
  var timer = [], prossimoTimer = 1;

  function Nodo(tipo) {
    this.nodeType = tipo;
    this.childNodes = [];
    this.parentNode = null;
  }
  Nodo.prototype.appendChild = function (n) {
    if (!n || typeof n !== "object" || !n.nodeType) {
      throw new TypeError("appendChild: argomento non è un nodo");
    }
    if (n.parentNode) {
      n.parentNode.removeChild(n);
    }
    n.parentNode = this;
    this.childNodes.push(n);
    return n;
  };
  Nodo.prototype.removeChild = function (n) {
    var i = this.childNodes.indexOf(n);
    if (i < 0) {
      throw new Error("removeChild: non è un figlio");
    }
    this.childNodes.splice(i, 1);
    n.parentNode = null;
    return n;
  };
  Object.defineProperty(Nodo.prototype, "firstChild", {get: function () { return this.childNodes[0] || null; }});
  Object.defineProperty(Nodo.prototype, "textContent", {
    get: function () {
      if (this.nodeType === 3) {
        return this.dato;
      }
      return this.childNodes.map(function (c) { return c.textContent; }).join("");
    },
    set: function (t) {
      if (this.nodeType === 3) {
        this.dato = String(t);
        return;
      }
      this.childNodes.forEach(function (c) { c.parentNode = null; });
      this.childNodes = [];
      if (t !== "" && t !== null && t !== undefined) {
        this.appendChild(new Testo(String(t)));
      }
    }
  });

  function Testo(t) {
    Nodo.call(this, 3);
    this.dato = t;
  }
  Testo.prototype = Object.create(Nodo.prototype);

  function ListaClassi(e) {
    this.e = e;
  }
  ListaClassi.prototype.elenco = function () {
    return (this.e.getAttribute("class") || "").split(/\s+/).filter(function (x) { return x; });
  };
  ListaClassi.prototype.contains = function (c) { return this.elenco().indexOf(c) >= 0; };
  ListaClassi.prototype.add = function (c) {
    var l = this.elenco();
    if (l.indexOf(c) < 0) {
      l.push(c);
    }
    this.e.setAttribute("class", l.join(" "));
  };
  ListaClassi.prototype.remove = function (c) {
    this.e.setAttribute("class", this.elenco().filter(function (x) { return x !== c; }).join(" "));
  };
  ListaClassi.prototype.toggle = function (c, forza) {
    var si = forza === undefined ? !this.contains(c) : !!forza;
    if (si) {
      this.add(c);
    } else {
      this.remove(c);
    }
    return si;
  };

  function Elemento(tag) {
    Nodo.call(this, 1);
    this.tagName = tag.toUpperCase();
    this.attributi = {};
    this.ascoltatori = {};
    this.style = {};
    this.classList = new ListaClassi(this);
    this.valore = "";
    this.spuntato = false;
    this.files = null;
  }
  Elemento.prototype = Object.create(Nodo.prototype);
  Elemento.prototype.setAttribute = function (k, v) { this.attributi[k] = String(v); };
  Elemento.prototype.getAttribute = function (k) {
    return Object.prototype.hasOwnProperty.call(this.attributi, k) ? this.attributi[k] : null;
  };
  Elemento.prototype.hasAttribute = function (k) { return Object.prototype.hasOwnProperty.call(this.attributi, k); };
  Elemento.prototype.removeAttribute = function (k) { delete this.attributi[k]; };
  Elemento.prototype.addEventListener = function (tipo, fn) {
    (this.ascoltatori[tipo] = this.ascoltatori[tipo] || []).push(fn);
  };
  Elemento.prototype.dispatchEvent = function (ev) {
    var self = this;
    ev.target = ev.target || self;
    (self.ascoltatori[ev.type] || []).slice().forEach(function (fn) { fn.call(self, ev); });
    return true;
  };
  Elemento.prototype.click = function () {
    if (this.disabled) {
      return;
    }
    if (this.tagName === "INPUT" && this.getAttribute("type") === "checkbox") {
      this.checked = !this.checked;
      this.dispatchEvent({type: "change", preventDefault: function () {}});
    }
    this.dispatchEvent({type: "click", preventDefault: function () {}});
  };
  Elemento.prototype.scrollIntoView = function () {};
  Elemento.prototype.select = function () {};
  Elemento.prototype.focus = function () {};
  function riflesso(nome) {
    Object.defineProperty(Elemento.prototype, nome, {
      get: function () { return this.hasAttribute(nome); },
      set: function (v) {
        if (v) {
          this.setAttribute(nome, "");
        } else {
          this.removeAttribute(nome);
        }
      }
    });
  }
  ["disabled", "hidden", "readonly", "open"].forEach(riflesso);
  Object.defineProperty(Elemento.prototype, "className", {
    get: function () { return this.getAttribute("class") || ""; },
    set: function (v) { this.setAttribute("class", v); }
  });
  Object.defineProperty(Elemento.prototype, "id", {
    get: function () { return this.getAttribute("id") || ""; },
    set: function (v) { this.setAttribute("id", v); }
  });
  Object.defineProperty(Elemento.prototype, "title", {
    get: function () { return this.getAttribute("title") || ""; },
    set: function (v) { this.setAttribute("title", v); }
  });
  Object.defineProperty(Elemento.prototype, "checked", {
    get: function () { return this.spuntato; },
    set: function (v) { this.spuntato = !!v; }
  });
  Object.defineProperty(Elemento.prototype, "value", {
    get: function () {
      if (this.tagName === "OPTION") {
        return this.hasAttribute("value") ? this.getAttribute("value") : this.textContent;
      }
      return this.valore;
    },
    set: function (v) { this.valore = String(v); }
  });
  Object.defineProperty(Elemento.prototype, "innerHTML", {
    get: function () { return this.textContent; },
    set: function (v) {
      if (v !== "") {
        throw new Error("DOM simulato: innerHTML si azzera soltanto");
      }
      this.textContent = "";
    }
  });
  Object.defineProperty(Elemento.prototype, "naturalWidth", {get: function () { return 0; }});
  Object.defineProperty(Elemento.prototype, "naturalHeight", {get: function () { return 0; }});

  /* selettori: "a b, c" con parti composte da tag, .classe, #id, [attr], [attr=valore] o [attr="valore"] */
  function parteSemplice(testo) {
    var p = {tag: null, classi: [], id: null, attributi: []};
    var re = /([A-Za-z][A-Za-z0-9-]*)|\.([A-Za-z0-9_-]+)|#([A-Za-z0-9_-]+)|\[([A-Za-z0-9_-]+)(?:=("?)([^\]"]*)\5)?\]/g;
    var m, consumati = 0;
    while ((m = re.exec(testo)) !== null) {
      if (m.index !== consumati) {
        throw new Error("DOM simulato: selettore non supportato: " + testo);
      }
      consumati = re.lastIndex;
      if (m[1]) {
        p.tag = m[1].toUpperCase();
      } else if (m[2]) {
        p.classi.push(m[2]);
      } else if (m[3]) {
        p.id = m[3];
      } else {
        p.attributi.push([m[4], m[6] === undefined ? null : m[6]]);
      }
    }
    if (consumati !== testo.length) {
      throw new Error("DOM simulato: selettore non supportato: " + testo);
    }
    return p;
  }
  function combacia(e, p) {
    if (e.nodeType !== 1 || (p.tag && e.tagName !== p.tag) || (p.id && e.id !== p.id)) {
      return false;
    }
    if (!p.classi.every(function (c) { return e.classList.contains(c); })) {
      return false;
    }
    return p.attributi.every(function (a) {
      return e.hasAttribute(a[0]) && (a[1] === null || e.getAttribute(a[0]) === a[1]);
    });
  }
  function combaciaCatena(e, catena, radice) {
    var i = catena.length - 1;
    if (!combacia(e, catena[i])) {
      return false;
    }
    var cur = e.parentNode;
    i -= 1;
    while (i >= 0 && cur && cur !== radice) {
      if (combacia(cur, catena[i])) {
        i -= 1;
      }
      cur = cur.parentNode;
    }
    return i < 0;
  }
  function discendenti(n, out) {
    n.childNodes.forEach(function (c) {
      if (c.nodeType === 1) {
        out.push(c);
        discendenti(c, out);
      }
    });
    return out;
  }
  function cerca(radice, selettore) {
    var catene = selettore.split(",").map(function (s) {
      return s.trim().split(/\s+/).map(parteSemplice);
    });
    return discendenti(radice, []).filter(function (e) {
      return catene.some(function (c) { return combaciaCatena(e, c, radice); });
    });
  }
  Nodo.prototype.querySelectorAll = function (s) { return cerca(this, s); };
  Nodo.prototype.querySelector = function (s) { return cerca(this, s)[0] || null; };

  function Documento() {
    Nodo.call(this, 9);
    this.documentElement = new Elemento("html");
    this.appendChild(this.documentElement);
    this.body = new Elemento("body");
    this.documentElement.appendChild(this.body);
    this.hidden = false;
    this.ascoltatori = {};
  }
  Documento.prototype = Object.create(Nodo.prototype);
  Documento.prototype.createElement = function (tag) { return new Elemento(tag); };
  Documento.prototype.createTextNode = function (t) { return new Testo(String(t)); };
  Documento.prototype.getElementById = function (id) {
    return discendenti(this, []).filter(function (e) { return e.id === id; })[0] || null;
  };
  Documento.prototype.addEventListener = Elemento.prototype.addEventListener;
  Documento.prototype.dispatchEvent = Elemento.prototype.dispatchEvent;

  var documento = new Documento();
  ["risultati", "sintesi"].forEach(function (id) {  // gli elementi di verifica.html
    var e = documento.createElement(id === "risultati" ? "table" : "div");
    e.id = id;
    documento.body.appendChild(e);
  });

  var ascoltatoriFinestra = {};
  root.document = documento;
  root.navigator = {userAgent: "DOM simulato (scripts/review_package_dom.js) su Duktape"};
  root.location = {hash: ""};
  root.scrollTo = function () {};
  root.confirm = function () { throw new Error("DOM simulato: confirm non previsto (l'autotest passa conferma)"); };
  root.alert = root.confirm;
  root.addEventListener = function (tipo, fn) { (ascoltatoriFinestra[tipo] = ascoltatoriFinestra[tipo] || []).push(fn); };
  root.setTimeout = function (fn, ms) {
    var id = prossimoTimer++;
    timer.push({id: id, fn: fn, ms: ms || 0});
    return id;
  };
  root.clearTimeout = function (id) {
    timer = timer.filter(function (t) { return t.id !== id; });
  };
  root.Image = function () {
    return documento.createElement("img");  // non carica nulla: onload e onerror non arrivano mai
  };
  root.Blob = function (parti, opzioni) {
    this.testo = parti.join("");
    this.type = opzioni && opzioni.type;
  };
  root.URL = {createObjectURL: function () { throw new Error("DOM simulato: scaricamento non previsto"); },
              revokeObjectURL: function () {}};
  root.__DOM = {
    simulato: true,
    svuotaTimer: function () {  // esegue i timer in sospeso, in ordine, finché ce ne sono (al più 1000 giri)
      for (var giri = 0; timer.length && giri < 1000; giri++) {
        var t = timer.shift();
        t.fn();
      }
      return timer.length;
    },
    timerInSospeso: function () { return timer.length; }
  };
})(this);
