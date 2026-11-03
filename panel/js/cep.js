/* Minimal bridge to the CEP runtime (replaces Adobe's CSInterface.js). */
(function (global) {
  var cep = global.__adobe_cep__;
  var isWin = (global.navigator.platform || "").toLowerCase().indexOf("win") === 0;

  global.Host = {
    available: !!cep,
    isWindows: isWin,

    /** Run ExtendScript in Premiere Pro, resolves with the string result. */
    evalScript: function (script) {
      return new Promise(function (resolve) {
        if (!cep) return resolve('{"ok":false,"error":"Not running inside Premiere Pro"}');
        cep.evalScript(script, function (res) { resolve(res); });
      });
    },

    /** Call a BAS.* function in host.jsx with a JSON argument, returns parsed JSON. */
    call: function (fn, arg) {
      var payload = arg === undefined ? "" : JSON.stringify(JSON.stringify(arg));
      // keep the script pure ASCII so non-English file names survive evalScript
      payload = payload.replace(/[\u007f-\uffff]/g, function (c) {
        return "\\u" + ("0000" + c.charCodeAt(0).toString(16)).slice(-4);
      });
      return this.evalScript("BAS." + fn + "(" + payload + ")").then(function (res) {
        try { return JSON.parse(res); }
        catch (e) { return { ok: false, error: "Premiere returned: " + res }; }
      });
    },

    extensionPath: function () {
      if (!cep) return "";
      var p = decodeURI(cep.getSystemPath("extension"));
      return isWin ? p.replace(/^file:\/\/\//, "") : p.replace(/^file:\/\//, "");
    },

    openURL: function (url) {
      if (global.cep && global.cep.util && global.cep.util.openURLInDefaultBrowser) {
        global.cep.util.openURLInDefaultBrowser(url);
      } else {
        global.open(url, "_blank");
      }
    },

    nodeRequire: function (name) {
      var r = (global.cep_node && global.cep_node.require) || global.require;
      return r ? r(name) : null;
    }
  };
})(window);
