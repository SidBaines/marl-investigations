# ruff: noqa: E501 -- the page's CSS and JavaScript are embedded verbatim as strings.
"""Render a snapshot as one self-contained page (claude.ai Artifact fragment or standalone).

The page carries its snapshot as JSON and renders it with plain JavaScript; when
served over http(s) it polls ``snapshot.json`` beside itself and re-renders on
newer data. The fragment follows the Artifact page contract (no doctype, html,
head or body tags; ``<title>`` then ``<style>`` first); ``standalone`` wraps the
same content in a complete document for opening from disk or ``http.server``.
"""

from __future__ import annotations

import html
import json
from typing import Any

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
    "&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600"
    '&display=swap">'
)

STYLE = r"""
:root {
  --ground: #eceff2; --surface: #fbfcfd; --surface-2: #f3f5f8; --ink: #16212d;
  --ink-2: #4b5967; --ink-3: #7a8896; --rule: #d3dae1; --grid: #e3e8ed;
  --accent: #2f55b8; --accent-soft: rgba(47, 85, 184, 0.13);
  --run: #16804f; --run-soft: #dcf1e6; --warn: #a35f00; --warn-soft: #f8ead3;
  --bad: #b3261e; --bad-soft: #f9e0dd; --pause: #5b57a6; --pause-soft: #e6e5f4;
  --done: #56636f; --done-soft: #e2e7ec;
  --series-1: #2a78d6; --series-2: #ba8416; --series-3: #3b6f30; --series-4: #d55181;
  --series-5: #4a3aa7; --series-6: #e34948;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --cond: "IBM Plex Sans Condensed", "Arial Narrow", "Roboto Condensed", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0e141a; --surface: #141c24; --surface-2: #1a242e; --ink: #e3e9ef;
    --ink-2: #a7b4c1; --ink-3: #74838f; --rule: #26323e; --grid: #1f2a35;
    --accent: #86a6ff; --accent-soft: rgba(134, 166, 255, 0.16);
    --run: #4cc790; --run-soft: rgba(76, 199, 144, 0.14); --warn: #e3a548;
    --warn-soft: rgba(227, 165, 72, 0.15); --bad: #f2796f; --bad-soft: rgba(242, 121, 111, 0.15);
    --pause: #a9a5f0; --pause-soft: rgba(169, 165, 240, 0.15); --done: #93a1ad;
    --done-soft: rgba(147, 161, 173, 0.14);
    --series-1: #3987e5; --series-2: #b98a24; --series-3: #2b7c37; --series-4: #d55181;
    --series-5: #9085e9; --series-6: #e66767;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --ground: #0e141a; --surface: #141c24; --surface-2: #1a242e; --ink: #e3e9ef;
  --ink-2: #a7b4c1; --ink-3: #74838f; --rule: #26323e; --grid: #1f2a35;
  --accent: #86a6ff; --accent-soft: rgba(134, 166, 255, 0.16);
  --run: #4cc790; --run-soft: rgba(76, 199, 144, 0.14); --warn: #e3a548;
  --warn-soft: rgba(227, 165, 72, 0.15); --bad: #f2796f; --bad-soft: rgba(242, 121, 111, 0.15);
  --pause: #a9a5f0; --pause-soft: rgba(169, 165, 240, 0.15); --done: #93a1ad;
  --done-soft: rgba(147, 161, 173, 0.14);
  --series-1: #3987e5; --series-2: #b98a24; --series-3: #2b7c37; --series-4: #d55181;
  --series-5: #9085e9; --series-6: #e66767;
  color-scheme: dark;
}
body { margin: 0; background: var(--ground); color: var(--ink); font: 14px/1.45 var(--sans); }
#app { max-width: 1180px; margin: 0 auto; padding-inline: 16px; padding-block: 20px 48px;
  display: grid; gap: 22px; }
h1, h2, h3 { font-family: var(--cond); font-weight: 600; margin: 0; text-wrap: balance; }
h1 { font-size: 22px; letter-spacing: 0.01em; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.09em; color: var(--ink-2); }
h3 { font-size: 16px; }
.num, .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.muted { color: var(--ink-3); }
.top { display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: 10px 24px;
  border-bottom: 1px solid var(--rule); padding-bottom: 14px; }
.top .sub { color: var(--ink-3); font-size: 12.5px; margin-top: 3px; }
.clock { display: flex; align-items: center; gap: 10px; font-size: 12.5px; color: var(--ink-2); }
.chip { display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 2px 9px;
  font: 500 12px/1.6 var(--cond); letter-spacing: 0.03em; white-space: nowrap; }
.s-running { color: var(--run); background: var(--run-soft); }
.s-stalled, .l-warn { color: var(--warn); background: var(--warn-soft); }
.s-stopped, .l-bad, .s-down { color: var(--bad); background: var(--bad-soft); }
.s-paused, .l-info { color: var(--pause); background: var(--pause-soft); }
.s-complete, .s-idle { color: var(--done); background: var(--done-soft); }
.chip svg { width: 9px; height: 9px; flex: none; }
.attention { display: flex; flex-wrap: wrap; gap: 8px; }
.attention .chip { white-space: normal; border-radius: 6px; padding: 5px 10px; font-size: 12.5px; }
.attention .path { font-family: var(--mono); font-size: 11.5px; opacity: 0.85; }
.panel { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; }
.section { display: grid; gap: 10px; }
.section-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; }
.facts { display: flex; flex-wrap: wrap; gap: 6px 22px; font-size: 12.5px; }
.facts dt { color: var(--ink-3); font-family: var(--cond); letter-spacing: 0.03em; }
.facts div { display: flex; gap: 6px; align-items: baseline; min-width: 0; }
.facts dd { margin: 0; font-family: var(--mono); font-size: 12px; overflow-wrap: anywhere; }
.compute { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
.gpu { display: grid; grid-template-columns: 70px 1fr auto; gap: 4px 12px; align-items: center;
  padding: 10px 14px; border-top: 1px solid var(--grid); }
.gpu:first-of-type { border-top: 0; }
.gpu .label { font: 600 13px var(--cond); }
.gpu .label small { display: block; font: 400 11px var(--sans); color: var(--ink-3); }
.meter { position: relative; height: 8px; background: var(--surface-2); border-radius: 4px; overflow: hidden;
  box-shadow: inset 0 0 0 1px var(--grid); }
.meter i { position: absolute; inset: 0 auto 0 0; background: var(--accent); border-radius: 4px; }
.meter.util i { background: var(--run); }
.gpu .vals { font-size: 12px; text-align: right; color: var(--ink-2); }
.panel-title { padding: 10px 14px 0; }
.server { padding: 10px 14px; border-top: 1px solid var(--grid); display: grid; gap: 4px; }
.server:first-of-type { border-top: 0; }
.server .line { display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: baseline; font-size: 12.5px; }
.server .url { font-family: var(--mono); font-size: 12px; overflow-wrap: anywhere; }
.study { display: grid; gap: 8px; }
.study-head { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: baseline; }
.study-head h3 { font-family: var(--mono); font-weight: 500; font-size: 14.5px; overflow-wrap: anywhere; }
details.run { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; }
details.run[open] { border-color: color-mix(in srgb, var(--accent) 35%, var(--rule)); }
details.run > summary { list-style: none; cursor: pointer; display: grid;
  grid-template-columns: 104px minmax(0, 2.2fr) minmax(150px, 1.6fr) 104px 64px 70px;
  gap: 6px 14px; align-items: center; padding: 10px 14px; border-radius: 8px; }
details.run > summary::-webkit-details-marker { display: none; }
details.run > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.run-name { min-width: 0; }
.run-name b { display: block; font: 500 13px var(--mono); overflow-wrap: anywhere; }
.run-name span { font-size: 11.5px; color: var(--ink-3); }
.prog { display: grid; gap: 4px; }
.prog .meter { height: 6px; }
.prog .meter.part i { background: color-mix(in srgb, var(--accent) 45%, transparent); }
.prog .txt { font-size: 11.5px; color: var(--ink-2); }
.cell { font-size: 12px; color: var(--ink-2); text-align: right; white-space: nowrap; }
.cell small { display: block; color: var(--ink-3); font: 11px var(--cond); letter-spacing: 0.04em;
  text-transform: uppercase; }
.detail { padding: 4px 14px 16px; display: grid; gap: 14px; border-top: 1px solid var(--grid); }
.multiples { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
.spark { background: var(--surface-2); border: 1px solid var(--grid); border-radius: 6px; padding: 8px 10px 6px;
  display: grid; gap: 2px; min-width: 0; }
.spark .k { font: 500 12px var(--cond); color: var(--ink-2); letter-spacing: 0.02em; overflow-wrap: anywhere; }
.spark .v { font: 500 17px var(--mono); font-variant-numeric: tabular-nums; }
.spark svg { width: 100%; height: 46px; display: block; }
.spark .range { font: 10.5px var(--mono); color: var(--ink-3); display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0 8px; }
.spark .range span { white-space: nowrap; }
.groups { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 330px), 1fr)); gap: 10px; min-width: 0; }
.groups-cap { font-size: 12px; color: var(--ink-3); }
.gchart { background: var(--surface-2); border: 1px solid var(--grid); border-radius: 6px; padding: 9px 12px 8px;
  display: grid; gap: 6px; min-width: 0; }
.gchart .t { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 0 10px; }
.gchart .t b { font: 600 14px/1.3 var(--cond); letter-spacing: 0.01em; }
.gchart .t span { font: 10.5px var(--mono); color: var(--ink-3); overflow-wrap: anywhere; }
.legend { display: flex; flex-wrap: wrap; gap: 2px 12px; margin: 0; padding: 0; list-style: none; font-size: 12px; color: var(--ink-2); }
.legend li { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.legend i { display: inline-block; width: 15px; border-top: 2px solid; border-radius: 1px; }
.legend li.avg { color: var(--ink); font-weight: 500; }
.legend li.avg i { border-top-width: 3px; }
.legend .num { color: var(--ink); font-size: 11.5px; }
.gplot { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 2px 6px; }
.gplot svg { width: 100%; height: 150px; display: block; }
.gplot .yax { display: flex; flex-direction: column; justify-content: space-between; text-align: right;
  font: 10.5px/12px var(--mono); color: var(--ink-3); }
.gplot .xax { grid-column: 2; display: flex; justify-content: space-between; gap: 8px; font: 10.5px var(--mono); color: var(--ink-3); }
.gplot .hit { fill: transparent; }
.gplot .hit:hover { fill: var(--ink); fill-opacity: 0.06; }
.hide-raw .g-raw { display: none; }
.toggle { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--ink-2); cursor: pointer; margin-left: auto; }
.toggle input { accent-color: var(--accent); margin: 0; }
.scroll { overflow-x: auto; }
table.grid { border-collapse: collapse; font-size: 12px; min-width: 100%; }
table.grid th, table.grid td { padding: 4px 10px; border-bottom: 1px solid var(--grid); text-align: right;
  white-space: nowrap; }
table.grid th { font: 600 11.5px var(--cond); color: var(--ink-3); letter-spacing: 0.04em; }
table.grid th:first-child, table.grid td:first-child { text-align: left; }
table.grid td { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.empty { padding: 28px 18px; text-align: center; color: var(--ink-2); }
.subhead { font: 600 12px var(--cond); color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.08em; }
details.more > summary { cursor: pointer; font: 500 12.5px var(--cond); color: var(--accent); }
details.more > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (max-width: 760px) {
  details.run > summary { grid-template-columns: 1fr auto; }
  details.run > summary .run-name { grid-column: 1 / -1; order: -1; }
  details.run > summary .prog { grid-column: 1 / -1; }
  .cell { text-align: left; }
  .gpu { grid-template-columns: 56px 1fr; }
  .gpu .vals { grid-column: 1 / -1; text-align: left; }
}
@media (prefers-reduced-motion: no-preference) {
  .meter i { transition: width 0.4s ease; }
}
"""

SCRIPT = r"""
(function () {
  "use strict";
  var el = document.getElementById("snapshot");
  var snap = JSON.parse(el.textContent);
  var app = document.getElementById("app");
  var openRuns = new Set();
  try { JSON.parse(localStorage.getItem("marli-dash-open") || "[]").forEach(function (p) { openRuns.add(p); }); } catch (e) {}
  var SVGNS = "http://www.w3.org/2000/svg";
  var showRaw = true;
  try { showRaw = localStorage.getItem("marli-dash-raw") !== "0"; } catch (e) {}
  function applyRaw() { app.classList.toggle("hide-raw", !showRaw); }
  applyRaw();
  var NAMES = { "credit/no_signal_share": "Share of groups with no learning signal (all playthroughs tied)" };

  function h(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (attrs[k] == null) continue;
      if (k === "text") node.textContent = attrs[k];
      else if (k === "cls") node.className = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
    for (var i = 2; i < arguments.length; i++) {
      var kid = arguments[i];
      if (kid == null || kid === false) continue;
      node.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    }
    return node;
  }
  function s(tag, attrs) {
    var node = document.createElementNS(SVGNS, tag);
    for (var k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }
  function fmt(v) {
    if (v == null || typeof v !== "number" || !isFinite(v)) return "–";
    var a = Math.abs(v);
    if (Number.isInteger(v) && a < 1e4) return String(v);
    if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return "0";
    return v.toPrecision(2);
  }
  function dur(sec) {
    if (sec == null || !isFinite(sec)) return "–";
    sec = Math.max(0, sec);
    if (sec < 90) return Math.round(sec) + "s";
    if (sec < 5400) return Math.round(sec / 60) + "m";
    if (sec < 172800) return (sec / 3600).toFixed(1) + "h";
    return (sec / 86400).toFixed(1) + "d";
  }
  function now() { return Date.now() / 1000; }
  function glyph(status) {
    var svg = s("svg", { viewBox: "0 0 10 10", "aria-hidden": "true" });
    var fill = "currentColor";
    if (status === "running") svg.appendChild(s("circle", { cx: 5, cy: 5, r: 4, fill: fill }));
    else if (status === "stalled") svg.appendChild(s("path", { d: "M5 1 9.2 9H.8z", fill: fill }));
    else if (status === "stopped" || status === "down") svg.appendChild(s("rect", { x: 1, y: 1, width: 8, height: 8, fill: fill }));
    else if (status === "paused") { svg.appendChild(s("rect", { x: 1.5, y: 1, width: 2.4, height: 8, fill: fill })); svg.appendChild(s("rect", { x: 6.1, y: 1, width: 2.4, height: 8, fill: fill })); }
    else svg.appendChild(s("path", { d: "M1.2 5.3 4 8 8.8 2", fill: "none", stroke: fill, "stroke-width": 1.8 }));
    return svg;
  }
  function pill(status, label) {
    var c = h("span", { cls: "chip s-" + status }, glyph(status), label || status);
    return c;
  }
  function meter(frac, cls) {
    var m = h("div", { cls: "meter" + (cls ? " " + cls : "") });
    var i = h("i");
    i.style.width = (Math.max(0, Math.min(1, frac || 0)) * 100).toFixed(1) + "%";
    m.appendChild(i);
    return m;
  }
  function label(key) {
    if (NAMES[key]) return NAMES[key];
    return key.replace(/^grades\/_system\//, "system · ").replace(/^grades\//, "").replace(/^learner\//, "")
      .replace(/_seconds$/, " (s)").split("/").join(" · ").replace(/_/g, " ");
  }

  function spark(key, pts) {
    var W = 200, H = 46, P = 5;
    var ys = pts.map(function (p) { return p[1]; });
    var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
    if (hi === lo) { var pad = Math.abs(hi) * 0.1 || 1; lo -= pad; hi += pad; }
    var x0 = pts[0][0], x1 = pts[pts.length - 1][0];
    function X(x) { return x1 === x0 ? W / 2 : ((x - x0) / (x1 - x0)) * (W - 2 * P) + P; }
    function Y(y) { return H - P - ((y - lo) / (hi - lo)) * (H - 2 * P); }
    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none", role: "img",
      "aria-label": label(key) + " over " + pts.length + " steps" });
    [0, 0.5, 1].forEach(function (f) {
      var y = P + f * (H - 2 * P);
      svg.appendChild(s("line", { x1: 0, x2: W, y1: y, y2: y, stroke: "var(--grid)", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }));
    });
    var line = pts.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1); }).join(" ");
    if (pts.length > 1) {
      svg.appendChild(s("path", { d: line + " L" + X(x1).toFixed(1) + " " + (H - P) + " L" + X(x0).toFixed(1) + " " + (H - P) + "Z", fill: "var(--accent-soft)", stroke: "none" }));
      svg.appendChild(s("path", { d: line, fill: "none", stroke: "var(--accent)", "stroke-width": 1.6, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round" }));
    }
    var last = pts[pts.length - 1];
    svg.appendChild(s("circle", { cx: X(last[0]), cy: Y(last[1]), r: 2.6, fill: "var(--accent)", stroke: "var(--surface-2)", "stroke-width": 1 }));
    return h("div", { cls: "spark" },
      h("div", { cls: "k", text: label(key) }),
      h("div", { cls: "v", text: fmt(last[1]) }),
      svg,
      h("div", { cls: "range" },
        h("span", { text: fmt(Math.min.apply(null, ys)) + " – " + fmt(Math.max.apply(null, ys)) }),
        h("span", { text: pts.length > 1 ? "steps " + x0 + "–" + x1 : "step " + x0 })));
  }

  function groupChart(group, smoothN) {
    var W = 300, H = 150, P = 6;
    var series = group.series || [];
    var all = [], steps = {};
    series.forEach(function (sr) {
      sr.points.forEach(function (p) { all.push(p[1]); steps[p[0]] = 1; });
      sr.smooth.forEach(function (p) { all.push(p[1]); });
    });
    var xs = Object.keys(steps).map(Number).sort(function (a, b) { return a - b; });
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    var ylo = lo, yhi = hi;
    if (yhi === ylo) { var pad = Math.abs(yhi) * 0.1 || 1; ylo -= pad; yhi += pad; }
    var x0 = xs[0], x1 = xs[xs.length - 1];
    function X(x) { return x1 === x0 ? W / 2 : ((x - x0) / (x1 - x0)) * (W - 2 * P) + P; }
    function Y(y) { return H - P - ((y - ylo) / (yhi - ylo)) * (H - 2 * P); }
    function path(pts) { return pts.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1); }).join(" "); }
    function color(sr) { return sr.agent === "_system" ? "var(--ink)" : "var(--series-" + ((sr.slot || 0) % 6 + 1) + ")"; }
    function dash(sr) { return sr.agent !== "_system" && sr.slot >= 6 ? "5 3" : null; }
    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none", role: "img",
      "aria-label": group.title + ": " + series.map(function (sr) { return sr.label; }).join(", ") + " over " + xs.length + " steps" });
    [0, 0.5, 1].forEach(function (f) {
      var y = P + f * (H - 2 * P);
      svg.appendChild(s("line", { x1: 0, x2: W, y1: y, y2: y, stroke: "var(--grid)", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }));
    });
    var base = { fill: "none", "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round", "stroke-linecap": "round" };
    function line(sr, pts, extra) {
      var attrs = Object.assign({ d: path(pts), stroke: color(sr) }, base, extra);
      if (dash(sr)) attrs["stroke-dasharray"] = dash(sr);
      return s("path", attrs);
    }
    if (smoothN > 1) series.forEach(function (sr) {
      if (sr.points.length > 1) svg.appendChild(line(sr, sr.points, { "stroke-width": 1, "stroke-opacity": 0.4, "class": "g-raw" }));
    });
    series.forEach(function (sr) {
      var avg = sr.agent === "_system";
      if (sr.smooth.length > 1) svg.appendChild(line(sr, sr.smooth, { "stroke-width": avg ? 3 : 2 }));
      var last = sr.smooth[sr.smooth.length - 1];
      var dx = X(last[0]).toFixed(1), dy = Y(last[1]).toFixed(1);
      svg.appendChild(s("path", { d: "M" + dx + " " + dy + " L" + dx + " " + dy, stroke: color(sr), "stroke-width": avg ? 7 : 6, "stroke-linecap": "round", "vector-effect": "non-scaling-stroke" }));
    });
    // One invisible column per step; its tooltip lists every line's value there.
    var lookup = series.map(function (sr) {
      var raw = {}, sm = {};
      sr.points.forEach(function (p) { raw[p[0]] = p[1]; });
      sr.smooth.forEach(function (p) { sm[p[0]] = p[1]; });
      return { label: sr.label, raw: raw, sm: sm };
    });
    xs.forEach(function (x, i) {
      var left = i ? (X(xs[i - 1]) + X(x)) / 2 : 0, right = i < xs.length - 1 ? (X(x) + X(xs[i + 1])) / 2 : W;
      var rect = s("rect", { x: left.toFixed(1), y: 0, width: Math.max(0.5, right - left).toFixed(1), height: H, "class": "hit" });
      var lines = ["step " + x];
      lookup.forEach(function (row) {
        if (!(x in row.raw)) return;
        lines.push(row.label + ": " + fmt(row.sm[x]) + (smoothN > 1 ? " (this step " + fmt(row.raw[x]) + ")" : ""));
      });
      var title = s("title", {});
      title.textContent = lines.join("\n");
      rect.appendChild(title);
      svg.appendChild(rect);
    });
    var legend = h("ul", { cls: "legend" });
    series.forEach(function (sr) {
      var sw = h("i");
      sw.style.borderTopColor = color(sr);
      if (dash(sr)) sw.style.borderTopStyle = "dashed";
      legend.appendChild(h("li", { cls: sr.agent === "_system" ? "avg" : null }, sw, sr.label + " ",
        h("span", { cls: "num", text: fmt(sr.smooth[sr.smooth.length - 1][1]) })));
    });
    return h("div", { cls: "gchart" },
      h("div", { cls: "t" }, h("b", { text: group.title }), h("span", { text: group.component })),
      legend,
      h("div", { cls: "gplot" },
        h("div", { cls: "yax", "aria-hidden": "true" }, h("span", { text: fmt(yhi) }), h("span", { text: fmt(ylo) })),
        svg,
        h("div", { cls: "xax" }, h("span", { text: "step " + x0 }), h("span", { text: xs.length > 1 ? "step " + x1 : "" }))));
  }

  function groupCharts(train) {
    var groups = (train && train.groups) || [];
    if (!groups.length) return null;
    var n = snap.smooth_steps || 1;
    var grid = h("div", { cls: "groups" });
    groups.forEach(function (g) { grid.appendChild(groupChart(g, n)); });
    var cap = n > 1
      ? "Solid lines: average of the last " + n + " steps. Faint lines: each step on its own. Thick line: the average over agents. Legend numbers are the latest solid-line values."
      : "Each step's value. Thick line: the average over agents. Legend numbers are the latest values.";
    return h("div", { style: "display:grid;gap:8px" }, h("div", { cls: "groups-cap", text: cap }), grid);
  }

  function facts(obj) {
    var dl = h("dl", { cls: "facts" });
    Object.keys(obj || {}).forEach(function (k) {
      var v = obj[k];
      if (v == null || v === "" || (typeof v === "object" && !Object.keys(v).length)) return;
      var text = typeof v === "object" ? Object.keys(v).map(function (kk) { return kk + "=" + (typeof v[kk] === "number" ? fmt(v[kk]) : v[kk]); }).join("  ") : (typeof v === "number" ? String(v) : String(v));
      dl.appendChild(h("div", null, h("dt", { text: k.replace(/_/g, " ") }), h("dd", { text: text })));
    });
    return dl.childNodes.length ? dl : null;
  }

  function gradeTable(grades) {
    var agents = Object.keys(grades || {});
    if (!agents.length) return null;
    agents.sort(function (a, b) { return a === "_system" ? 1 : b === "_system" ? -1 : a < b ? -1 : 1; });
    var comps = {};
    agents.forEach(function (a) { Object.keys(grades[a]).forEach(function (c) { comps[c] = 1; }); });
    var head = h("tr", null, h("th", { text: "component" }));
    agents.forEach(function (a) { head.appendChild(h("th", { text: a === "_system" ? "system" : a })); });
    var body = h("tbody");
    Object.keys(comps).sort().forEach(function (c) {
      var tr = h("tr", null, h("td", { text: c }));
      agents.forEach(function (a) { tr.appendChild(h("td", { text: fmt(grades[a][c]) })); });
      body.appendChild(tr);
    });
    return h("div", { cls: "scroll" }, h("table", { cls: "grid" }, h("thead", null, head), body));
  }

  function trainDetail(train) {
    var box = h("div", { cls: "detail-train" });
    var nodes = [];
    var cur = train.current_step;
    if (cur) {
      var frac = cur.episodes_total ? cur.episodes_done / cur.episodes_total : 0;
      nodes.push(h("div", { cls: "prog" },
        h("div", { cls: "subhead", text: "step " + cur.step + " sampling" }),
        meter(frac, "part"),
        h("div", { cls: "txt num", text: cur.episodes_done + (cur.episodes_total ? " / " + cur.episodes_total : "") + " episodes" + (cur.failures ? " · " + cur.failures + " failed" : "") + (cur.rate ? " · " + fmt(cur.rate) + " ep/min" : "") })));
    }
    var keys = (train.key_curves || []).filter(function (k) { return train.curves[k] && train.curves[k].length; });
    if (keys.length) {
      var grid = h("div", { cls: "multiples" });
      keys.forEach(function (k) { grid.appendChild(spark(k, train.curves[k])); });
      nodes.push(grid);
    } else if (!cur) {
      nodes.push(h("div", { cls: "muted", text: "No completed steps yet." }));
    }
    var rest = Object.keys(train.curves || {}).filter(function (k) { return keys.indexOf(k) < 0; }).sort();
    if (rest.length) {
      var body = h("tbody");
      rest.forEach(function (k) {
        var pts = train.curves[k];
        var first = pts[0][1], last = pts[pts.length - 1][1];
        body.appendChild(h("tr", null, h("td", { text: label(k) }), h("td", { text: fmt(first) }), h("td", { text: fmt(last) }), h("td", { text: String(pts.length) })));
      });
      nodes.push(h("details", { cls: "more" }, h("summary", { text: "All " + rest.length + " other curves" }),
        h("div", { cls: "scroll" }, h("table", { cls: "grid" }, h("thead", null, h("tr", null, h("th", { text: "curve" }), h("th", { text: "first" }), h("th", { text: "latest" }), h("th", { text: "steps" }))), body))));
    }
    nodes.forEach(function (n) { box.appendChild(n); });
    box.style.display = "grid"; box.style.gap = "12px";
    return box;
  }

  function runRow(run) {
    var d = h("details", { cls: "run" });
    if (openRuns.has(run.path) || (run.kind === "train rl" && run.status === "running" && !openRuns.has("!" + run.path))) d.open = true;
    d.addEventListener("toggle", function () {
      if (d.open) { openRuns.add(run.path); openRuns.delete("!" + run.path); }
      else { openRuns.delete(run.path); openRuns.add("!" + run.path); }
      try { localStorage.setItem("marli-dash-open", JSON.stringify(Array.from(openRuns))); } catch (e) {}
    });
    var p = run.progress || {};
    var frac = p.total ? p.done / p.total : (run.status === "complete" ? 1 : 0);
    var ptxt = p.done != null ? fmt(p.done) + (p.total ? " / " + fmt(p.total) : "") + " " + (p.unit || "") : "–";
    if (p.failures) ptxt += " · " + fmt(p.failures) + " failed";
    var updated = run.updated_ts ? dur(now() - run.updated_ts) + " ago" : "–";
    var sum = h("summary", null,
      h("div", null, pill(run.status)),
      h("div", { cls: "run-name" }, h("b", { text: run.name }), h("span", { text: run.kind + (run.config_hash ? " · " + run.config_hash : "") })),
      h("div", { cls: "prog" }, meter(frac), h("div", { cls: "txt num", text: ptxt })),
      h("div", { cls: "cell num" }, h("small", { text: "rate" }), run.rate ? fmt(run.rate.value) + " " + run.rate.unit : "–"),
      h("div", { cls: "cell num" }, h("small", { text: "eta" }), run.eta_s != null ? dur(run.eta_s) : "–"),
      h("div", { cls: "cell num" }, h("small", { text: "updated" }), updated));
    d.appendChild(sum);
    var detail = h("div", { cls: "detail" });
    var gc = run.train ? groupCharts(run.train) : null;
    if (gc) detail.appendChild(gc);
    var f = facts(Object.assign({ path: run.path }, run.facts || {}));
    if (f) detail.appendChild(f);
    if (run.train) detail.appendChild(trainDetail(run.train));
    if (run.eval) {
      var e = run.eval;
      var ef = facts({ "generated tokens": e.gen_tokens, "tokens / episode": e.gen_tokens_per_episode, "LLM calls": e.calls, "paused at tasks": e.paused_at_tasks, "unparsed rows": e.unparsed_rows || null });
      if (ef) detail.appendChild(ef);
      var gt = gradeTable(e.grades);
      if (gt) { detail.appendChild(h("div", { cls: "subhead", text: "Mean grade components (ok episodes)" })); detail.appendChild(gt); }
    }
    if (run.collect_error) detail.appendChild(h("div", { cls: "muted", text: "Summary failed: " + run.collect_error }));
    d.appendChild(detail);
    return d;
  }

  function compute() {
    var wrap = h("div", { cls: "compute" });
    if (snap.gpus && snap.gpus.length) {
      var panel = h("div", { cls: "panel" });
      snap.gpus.forEach(function (g) {
        var memFrac = g.mem_total_mib ? g.mem_used_mib / g.mem_total_mib : 0;
        panel.appendChild(h("div", { cls: "gpu" },
          h("div", { cls: "label" }, "GPU " + g.index, h("small", { text: g.name.replace(/^NVIDIA /, "") })),
          h("div", { style: "display:grid;gap:5px" }, meter(memFrac), meter(g.util / 100, "util")),
          h("div", { cls: "vals num", text: fmt(g.mem_used_mib / 1024) + " / " + fmt(g.mem_total_mib / 1024) + " GiB · " + Math.round(g.util) + "%" })));
      });
      wrap.appendChild(panel);
    }
    if (snap.servers && snap.servers.length) {
      var sp = h("div", { cls: "panel" });
      snap.servers.slice().sort(function (a, b) { return (b.up ? 1 : 0) - (a.up ? 1 : 0); }).forEach(function (sv) {
        var m = sv.metrics || {};
        var bits = [];
        if (sv.metrics) {
          bits.push(fmt(m.running) + " running", fmt(m.waiting) + " waiting");
          if (m.kv_usage != null) bits.push("KV " + Math.round(m.kv_usage * 100) + "%");
          if (m.gen_tokens_per_s != null) bits.push(fmt(m.gen_tokens_per_s) + " gen tok/s");
          if (m.spec_accept_rate != null) bits.push("spec accept " + Math.round(m.spec_accept_rate * 100) + "%");
          if (m.prefix_hit_rate != null) bits.push("prefix hit " + Math.round(m.prefix_hit_rate * 100) + "%");
        }
        sp.appendChild(h("div", { cls: "server" },
          h("div", { cls: "line" }, pill(sv.up ? "running" : "down", sv.up ? "serving" : "down"), h("span", { cls: "url", text: sv.base_url || "?" }), h("span", { cls: "muted", text: (sv.models || []).join(", ") + (sv.n_adapters ? " · " + sv.n_adapters + " adapters" : "") })),
          h("div", { cls: "line num muted", text: bits.join(" · ") || sv.path }),
          bits.length ? h("div", { cls: "line mono muted", style: "font-size:11px", text: sv.path }) : null));
      });
      wrap.appendChild(sp);
    }
    return wrap.childNodes.length ? wrap : null;
  }

  function annotations(a) {
    if (a == null) return null;
    if (Array.isArray(a) && a.length && typeof a[0] === "object") {
      var cols = Object.keys(a.reduce(function (acc, r) { Object.keys(r || {}).forEach(function (k) { acc[k] = 1; }); return acc; }, {}));
      var head = h("tr"); cols.forEach(function (c) { head.appendChild(h("th", { text: c })); });
      var body = h("tbody");
      a.forEach(function (r) { var tr = h("tr"); cols.forEach(function (c) { var v = r[c]; tr.appendChild(h("td", { text: v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v) })); }); body.appendChild(tr); });
      return h("div", { cls: "scroll panel" }, h("table", { cls: "grid" }, h("thead", null, head), body));
    }
    if (typeof a === "object") {
      // Scalars as facts; lists of records (e.g. pods) as their own small tables.
      var scalars = {}, blocks = [];
      Object.keys(a).forEach(function (k) {
        var v = a[k];
        if (Array.isArray(v) && v.length && typeof v[0] === "object") {
          blocks.push(h("section", { cls: "section" }, h("h2", { text: k }), annotations(v)));
        } else scalars[k] = v;
      });
      var wrap = h("div", { cls: "annotations" });
      blocks.forEach(function (b) { wrap.appendChild(b); });
      if (Object.keys(scalars).length) wrap.appendChild(facts(scalars));
      return wrap;
    }
    return h("div", { cls: "muted", text: String(a) });
  }

  function render() {
    var age = now() - snap.generated_ts;
    var ageLevel = age < 2.5 * (snap.refresh_s || 60) ? "running" : age < 1800 ? "stalled" : "stopped";
    var ageChip = h("span", { cls: "chip s-" + ageLevel, id: "age" }, glyph(ageLevel), "data " + dur(age) + " old");
    var gen = new Date(snap.generated_ts * 1000);
    var top = h("header", { cls: "top" },
      h("div", null, h("h1", { text: snap.title }), h("div", { cls: "sub mono", text: snap.host + " · " + (snap.roots || []).join(", ") })),
      h("div", { cls: "clock" }, h("span", { cls: "num", text: "snapshot " + gen.toISOString().slice(0, 19).replace("T", " ") + " UTC" }), ageChip));
    var nodes = [top];
    var ann = annotations(snap.annotations);
    if (ann) nodes.push(ann);
    var att = h("div", { cls: "attention" });
    (snap.attention || []).forEach(function (a) {
      att.appendChild(h("span", { cls: "chip l-" + a.level }, glyph(a.level === "bad" ? "stopped" : a.level === "warn" ? "stalled" : "paused"),
        h("span", null, a.text + " ", a.path ? h("span", { cls: "path", text: a.path }) : null)));
    });
    if (!att.childNodes.length) att.appendChild(h("span", { cls: "chip s-running" }, glyph("running"), "Nothing needs attention"));
    nodes.push(h("section", { cls: "section" }, h("h2", { text: "Attention" }), att));
    var comp = compute();
    if (comp) nodes.push(h("section", { cls: "section" }, h("h2", { text: "Compute" }), comp));
    var runs = snap.runs || [];
    var runsSec = h("section", { cls: "section" }, h("div", { cls: "section-head" }, h("h2", { text: "Runs" }),
      h("span", { cls: "muted num", style: "font-size:12px", text: Object.keys(snap.counts || {}).map(function (k) { return snap.counts[k] + " " + k; }).join(" · ") })));
    if (runs.some(function (r) { return r.train && r.train.groups && r.train.groups.length; })) {
      var cb = h("input", { type: "checkbox" });
      cb.checked = showRaw;
      cb.addEventListener("change", function () {
        showRaw = cb.checked;
        try { localStorage.setItem("marli-dash-raw", showRaw ? "1" : "0"); } catch (e) {}
        applyRaw();
      });
      runsSec.firstChild.appendChild(h("label", { cls: "toggle" }, cb, "show raw per-step values"));
    }
    if (!runs.length) {
      runsSec.appendChild(h("div", { cls: "panel empty", text: "No run directories under " + (snap.roots || []).join(", ") + " yet. A run appears once a marli verb writes its .marli/run.json." }));
    }
    var rank = { running: 0, stalled: 1, stopped: 2, paused: 3, complete: 4 };
    var studies = {};
    runs.forEach(function (r) { (studies[r.study] = studies[r.study] || []).push(r); });
    Object.keys(studies).sort(function (a, b) {
      function best(k) { return Math.min.apply(null, studies[k].map(function (r) { return rank[r.status] == null ? 5 : rank[r.status]; })); }
      return best(a) - best(b) || (a < b ? -1 : 1);
    }).forEach(function (name) {
      var list = studies[name].slice().sort(function (a, b) { return (rank[a.status] - rank[b.status]) || ((b.updated_ts || 0) - (a.updated_ts || 0)); });
      var counts = {};
      list.forEach(function (r) { counts[r.status] = (counts[r.status] || 0) + 1; });
      var head = h("div", { cls: "study-head" }, h("h3", { text: name }));
      Object.keys(counts).forEach(function (k) { head.appendChild(pill(k, counts[k] + " " + k)); });
      var st = h("div", { cls: "study" }, head);
      list.forEach(function (r) { st.appendChild(runRow(r)); });
      runsSec.appendChild(st);
    });
    nodes.push(runsSec);
    while (app.firstChild) app.removeChild(app.firstChild);
    nodes.forEach(function (n) { app.appendChild(n); });
  }

  render();
  setInterval(function () {
    var chip = document.getElementById("age");
    if (!chip) return;
    var age = now() - snap.generated_ts;
    chip.lastChild.textContent = "data " + dur(age) + " old";
  }, 15000);
  if (/^https?:$/.test(location.protocol)) {
    setInterval(function () {
      fetch("snapshot.json?t=" + Date.now(), { cache: "no-store" }).then(function (r) { return r.ok ? r.json() : null; })
        .then(function (next) { if (next && next.generated_ts > snap.generated_ts) { snap = next; render(); } })
        .catch(function () {});
    }, Math.max(10, snap.refresh_s || 60) * 1000);
  }
})();
"""


def _embed(snapshot: dict[str, Any]) -> str:
    # "<" never appears raw inside the script element, so no </script> breakout.
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )


def _parts(snapshot: dict[str, Any]) -> tuple[str, str]:
    title = html.escape(str(snapshot.get("title") or "marli runs"))
    head = f"<title>{title}</title>\n<style>{STYLE}</style>\n{FONTS}\n"
    body = (
        '<main id="app" aria-live="polite"></main>\n'
        f'<script type="application/json" id="snapshot">{_embed(snapshot)}</script>\n'
        f"<script>{SCRIPT}</script>\n"
    )
    return head, body


def render(snapshot: dict[str, Any], *, standalone: bool = False) -> str:
    """The page for ``snapshot``: an Artifact fragment, or a complete HTML document."""
    head, body = _parts(snapshot)
    if not standalone:
        return head + body
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{head}</head>\n<body>\n{body}</body>\n</html>\n"
    )
