const pptxgen = require("pptxgenjs");
const path = require("path");

// Palette: an inspection-cell stack lamp. Charcoal shop-floor ground, amber as
// the single accent, and green/red reserved *only* for actual verdicts so the
// colours mean the same thing on every slide that they mean on the machine.
const BG      = "14181C";
const CARD    = "1E242A";
const FG      = "EDEFF2";
const MUT     = "97A2AD";
const AMBER   = "E8A317";
const GREEN   = "3FB950";
const RED     = "F85149";

const H = "Arial";
const B = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5 in
pres.author = "Jash Pasad";
pres.company = "IIT Gandhinagar";
pres.subject = "Snapdragon AI Lab Build & Present Challenge 2026";
pres.title = "NIRIKSHAK";

const W = 13.3, HT = 7.5, M = 0.7;

function slide(dark = true) {
  const s = pres.addSlide();
  s.background = { color: dark ? BG : "FFFFFF" };
  return s;
}

function title(s, text, sub) {
  s.addText(text, {
    x: M, y: 0.48, w: W - 2 * M, h: 0.62, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 32, bold: true, color: FG,
  });
  if (sub) {
    s.addText(sub, {
      x: M, y: 1.12, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 14, color: MUT,
    });
  }
}

function card(s, x, y, w, h, fill = CARD) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill }, rectRadius: 0.08, line: { color: fill },
  });
}

function stat(s, x, y, w, value, label, colour = FG, sz = 40) {
  s.addText(value, {
    x, y, w, h: 0.72, isTextBox: true, margin: 0,
    fontFace: H, fontSize: sz, bold: true, color: colour,
  });
  s.addText(label, {
    x, y: y + 0.72, w, h: 0.5, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 11.5, color: MUT,
  });
}

function footer(s, n) {
  s.addText("NIRIKSHAK  ·  Jash Pasad, IIT Gandhinagar  ·  Snapdragon AI Lab Build & Present Challenge", {
    x: M, y: HT - 0.52, w: 9.5, h: 0.3, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 9.5, color: "5A646E",
  });
  s.addText(String(n), {
    x: W - M - 0.6, y: HT - 0.52, w: 0.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 9.5, color: "5A646E", align: "right",
  });
}

/* ── 1 · Title ─────────────────────────────────────────────────────────── */
{
  const s = slide();
  s.addText("NIRIKSHAK", {
    x: M, y: 2.25, w: 9, h: 1.1, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 62, bold: true, color: FG, charSpacing: 3,
  });
  s.addText("निरीक्षक  ·  “the inspector”", {
    x: M, y: 3.32, w: 9, h: 0.42, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 17, color: AMBER,
  });
  s.addText("Zero-shot visual quality inspection on Snapdragon.", {
    x: M, y: 4.0, w: 10.5, h: 0.45, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 21, color: FG,
  });
  s.addText(
    "An HP Snapdragon PC and an Arduino UNO Q become a factory inspection cell that learns a new part\n" +
    "from 20 good samples. No labels. No training. No cloud.",
    { x: M, y: 4.52, w: 11, h: 0.8, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 14.5, color: MUT, lineSpacing: 22 });
  s.addText("Jash Pasad", {
    x: M, y: 5.72, w: 9, h: 0.36, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 17, bold: true, color: FG,
  });
  s.addText("B.Tech · Indian Institute of Technology Gandhinagar", {
    x: M, y: 6.08, w: 9, h: 0.3, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 13, color: MUT,
  });
  s.addText("Snapdragon® AI Lab Build & Present Challenge  ·  Individual submission  ·  github.com/jashpasad/NIRIKSHAK", {
    x: M, y: 6.46, w: 11.5, h: 0.3, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 11, color: "5A646E",
  });
  s.addNotes("NIRIKSHAK is Hindi for 'the inspector'. One line to remember: it learns a new part from twenty good samples, with no labels, no training and no cloud.");
}

/* ── 2 · Problem ───────────────────────────────────────────────────────── */
{
  const s = slide();
  title(s, "Almost no Indian MSME inspects automatically",
           "6.3 crore units \u2014 and it is not because they don't want to.");

  const items = [
    ["₹15–40 lakh", "per machine-vision line",
     "Cognex, Keyence, Omron. More than the annual profit of most units that would benefit from one.", AMBER],
    ["8 defects/day", "unlabelled, un-enumerated",
     "A workshop at 400 parts/day and 2% rejects takes three years to collect a training set — and still cannot catch a defect type it has never seen.", AMBER],
    ["300 ms", "cloud round trip",
     "A conveyor does not wait. Shop floors lack reliable uplinks. And a job shop is contractually barred from uploading its customer's part geometry anywhere.", AMBER],
  ];
  let x = M;
  const cw = (W - 2 * M - 0.6) / 3;
  items.forEach(([big, small, body, col]) => {
    card(s, x, 1.85, cw, 3.5);
    s.addText(big, { x: x + 0.3, y: 2.12, w: cw - 0.6, h: 0.6, isTextBox: true, margin: 0,
      fontFace: H, fontSize: 27, bold: true, color: col });
    s.addText(small, { x: x + 0.3, y: 2.75, w: cw - 0.6, h: 0.32, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 12, color: MUT });
    s.addText(body, { x: x + 0.3, y: 3.22, w: cw - 0.6, h: 1.9, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13, color: FG, lineSpacing: 19 });
    x += cw + 0.3;
  });

  s.addText("So the market is stuck between “₹40 lakh” and “nothing”, and picks nothing.", {
    x: M, y: 5.75, w: W - 2 * M, h: 0.5, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 19, italic: true, color: AMBER,
  });
  footer(s, 2);
  s.addNotes("The key point is the middle card. The standard AI answer assumes a labelled defect dataset that will never exist at this scale.");
}

/* ── 3 · The reframe ───────────────────────────────────────────────────── */
{
  const s = slide();
  title(s, "The reframe");

  card(s, M, 1.75, W - 2 * M, 1.65);
  s.addText("A defect is not a thing you learn to recognise.\n" +
            "A defect is a region that does not look like anything you saw on a good part.", {
    x: M + 0.45, y: 1.95, w: W - 2 * M - 0.9, h: 1.25, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 22, color: FG, lineSpacing: 34,
  });

  s.addText("Model normality, not defects. Every factory already has that training data — it is the parts they ship.", {
    x: M, y: 3.65, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 15, color: AMBER,
  });

  const rows = [
    ["No labels", "Good parts need none."],
    ["No defect dataset", "We never model defects at all."],
    ["Unknown defects caught", "Anything unlike normal is anomalous, by construction."],
    ["Fits on a laptop", "Indexing, not gradient descent. Enrolment takes 5 seconds."],
  ];
  let y = 4.3;
  rows.forEach(([k, v]) => {
    s.addShape(pres.ShapeType.ellipse, { x: M + 0.05, y: y + 0.09, w: 0.16, h: 0.16,
      fill: { color: GREEN }, line: { color: GREEN } });
    s.addText(k, { x: M + 0.42, y, w: 2.9, h: 0.34, isTextBox: true, margin: 0,
      fontFace: H, fontSize: 14, bold: true, color: FG });
    s.addText(v, { x: M + 3.4, y, w: 8.2, h: 0.34, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13.5, color: MUT });
    y += 0.52;
  });
  footer(s, 3);
  s.addNotes("This inversion is what removes every constraint on the previous slide at once.");
}

/* ── 4 · Architecture ──────────────────────────────────────────────────── */
{
  const s = slide();
  title(s, "Four tiers, two chips", "Compute proportional to uncertainty");

  const tiers = [
    ["TIER 0", "GATE", "Arduino UNO Q\nDragonwing QRB2210", "~2 ms", "every camera frame", MUT],
    ["TIER 1", "SCREEN", "Hexagon NPU\npatch k-NN vs memory bank", "~10 ms", "every part", GREEN],
    ["TIER 2", "EXPLAIN", "Hexagon NPU\nQwen3-VL-4B", "~400 ms", "only the uncertainty band", AMBER],
    ["TIER 3", "REASON", "Hexagon NPU\nQwen3-4B", "~25 s", "once per shift", RED],
  ];
  let x = M;
  const cw = (W - 2 * M - 0.75) / 4;
  tiers.forEach(([t, name, where, cost, when, col]) => {
    card(s, x, 1.9, cw, 3.35);
    s.addText(t, { x: x + 0.28, y: 2.12, w: cw - 0.5, h: 0.28, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 10.5, bold: true, color: col, charSpacing: 1.5 });
    s.addText(name, { x: x + 0.28, y: 2.42, w: cw - 0.5, h: 0.45, isTextBox: true, margin: 0,
      fontFace: H, fontSize: 22, bold: true, color: FG });
    s.addText(where, { x: x + 0.28, y: 2.95, w: cw - 0.5, h: 0.9, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 12, color: MUT, lineSpacing: 17 });
    s.addText(cost, { x: x + 0.28, y: 3.95, w: cw - 0.5, h: 0.45, isTextBox: true, margin: 0,
      fontFace: H, fontSize: 21, bold: true, color: col });
    s.addText(when, { x: x + 0.28, y: 4.45, w: cw - 0.5, h: 0.6, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 12, color: MUT, lineSpacing: 16 });
    x += cw + 0.25;
  });

  s.addText("At 97% yield, Tier 2 fires on ~4% of parts. Mean cost is 10 + 0.04 × 400 ≈ 26 ms per part, not 400 ms.", {
    x: M, y: 5.5, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 15, color: FG,
  });
  s.addText("A 6× throughput gain from ordering the models — without making any single model faster.", {
    x: M, y: 5.95, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 16, bold: true, color: AMBER,
  });
  footer(s, 4);
  s.addNotes("This is the systems contribution. 99% of parts never touch a 4B-parameter model.");
}

/* ── 5 · Economics chart ───────────────────────────────────────────────── */
{
  const s = slide();
  title(s, "The gating gain grows as the line gets better",
           "Measured Tier-1 latency, modelled against a 400 ms Tier-2 VLM");

  s.addChart(pres.ChartType.bar, [
    { name: "Effective ms per part", labels: ["90% yield", "95%", "97%", "99%"],
      values: [95.6, 75.8, 67.9, 60.0] },
  ], {
    x: M, y: 1.85, w: 7.6, h: 4.1,
    barDir: "col", chartColors: [AMBER],
    showTitle: true, title: "Effective cost per part (ms)", titleColor: FG,
    titleFontFace: H, titleFontSize: 13,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: FG,
    dataLabelFontFace: B, dataLabelFontSize: 11,
    showLegend: false,
    catAxisLabelColor: MUT, valAxisLabelColor: MUT,
    catAxisLabelFontFace: B, valAxisLabelFontFace: B,
    catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
    valGridLine: { color: "2A3138", size: 1 }, catGridLine: { style: "none" },
    valAxisMaxVal: 120,
  });

  card(s, 8.6, 1.85, W - M - 8.6, 4.1);
  s.addText("Without the cascade", { x: 8.95, y: 2.1, w: 3.4, h: 0.3, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 11.5, color: MUT });
  s.addText("400 ms", { x: 8.95, y: 2.4, w: 3.4, h: 0.55, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 34, bold: true, color: RED });
  s.addText("every part, ~2.5 parts/sec, NPU pinned all shift", {
    x: 8.95, y: 2.98, w: 3.4, h: 0.6, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 12, color: MUT, lineSpacing: 16 });

  s.addText("With the cascade, at 97%", { x: 8.95, y: 3.85, w: 3.4, h: 0.3, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 11.5, color: MUT });
  s.addText("67.9 ms", { x: 8.95, y: 4.15, w: 3.4, h: 0.55, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 34, bold: true, color: GREEN });
  s.addText("883 parts/min sustainable  ·  5.9× faster", {
    x: 8.95, y: 4.73, w: 3.4, h: 0.6, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 12, color: MUT, lineSpacing: 16 });

  s.addText("Tier-1 figure is measured on the pessimistic path: CPU reference descriptor, no NPU.", {
    x: M, y: 6.15, w: W - 2 * M, h: 0.35, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 11.5, italic: true, color: "5A646E" });
  footer(s, 5);
}

/* ── 6 · Results ───────────────────────────────────────────────────────── */
{
  const s = slide();
  title(s, "Results", "24 enrolment samples · 40 held-out good parts · 40 defective across 5 types");

  const stats = [
    ["1.000", "image AUROC", GREEN],
    ["0%", "escape rate", GREEN],
    ["0 / 40", "false rejects", GREEN],
    ["40 / 40", "localised", GREEN],
    ["5.3 s", "enrolment", AMBER],
  ];
  let x = M;
  const cw = (W - 2 * M - 0.8) / 5;
  stats.forEach(([v, l, c]) => { stat(s, x, 1.85, cw, v, l, c, 34); x += cw + 0.2; });

  s.addImage({ path: path.join(__dirname, "..", "docs", "images", "results_grid.png"),
    x: M, y: 3.3, w: 8.55, h: 2.6 });

  card(s, 9.5, 3.3, W - M - 9.5, 2.6);
  s.addText("Per defect type", { x: 9.8, y: 3.5, w: 2.9, h: 0.3, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 11, color: MUT, charSpacing: 0.8 });
  const per = ["scratch", "dent", "chip", "contamination", "discolouration"];
  let y = 3.76;
  per.forEach(k => {
    s.addText(k, { x: 9.8, y, w: 2.0, h: 0.28, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 12, color: FG });
    s.addText("8 / 8", { x: 11.7, y, w: 0.9, h: 0.28, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 12, bold: true, color: GREEN, align: "right" });
    y += 0.30;
  });
  s.addText("Safe threshold window [0.78, 2.06] σ.\nWe ship 1.50 — the centre.", {
    x: 9.8, y: 5.36, w: 2.8, h: 0.48, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 10, italic: true, color: MUT, lineSpacing: 13 });

  s.addText("Input · anomaly heatmap with localised regions · verdict. The system was shown only good parts.", {
    x: M, y: 6.05, w: 8.55, h: 0.35, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 11, italic: true, color: "5A646E" });
  footer(s, 6);
}

/* ── 7 · Honesty ───────────────────────────────────────────────────────── */
{
  const s = slide();
  title(s, "What that 1.000 does and does not mean");

  card(s, M, 1.75, W - 2 * M, 1.5);
  s.addText("We generated that data, so we do not get to be impressed by it.", {
    x: M + 0.45, y: 2.05, w: W - 2 * M - 0.9, h: 0.9, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 24, color: AMBER });

  const cols = [
    ["What it proves", [
      "The pipeline is correct and reproducible on any machine.",
      "The generator injects the nuisance variation that normally defeats this method: lighting gradient, ±4 px shift, ±1.4° rotation, sensor noise.",
      "The part has a brushed finish whose parallel streaks are visually confusable with scratches — the hard case is deliberately present.",
    ], GREEN],
    ["What it does not prove", [
      "Field accuracy. Synthetic data is easier than reality.",
      "The external reference that carries weight is MVTec AD, where published PatchCore-class methods reach ~0.99 AUROC.",
      "Our evaluator already accepts MVTec's layout unchanged. Running it, plus a real camera trial, is the next round.",
    ], RED],
  ];
  let x = M;
  const cw = (W - 2 * M - 0.4) / 2;
  cols.forEach(([h, lines, col]) => {
    s.addText(h, { x, y: 3.55, w: cw, h: 0.35, isTextBox: true, margin: 0,
      fontFace: H, fontSize: 15, bold: true, color: col });
    s.addText(lines.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < lines.length - 1 } })), {
      x, y: 4.0, w: cw - 0.2, h: 2.0, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13, color: FG, lineSpacing: 19, paraSpaceAfter: 8 });
    x += cw + 0.4;
  });
  footer(s, 7);
  s.addNotes("Volunteering this is the point. A judge will find it anyway; better that we found it first.");
}

/* ── 8 · Engineering ───────────────────────────────────────────────────── */
{
  const s = slide();
  title(s, "Four findings, all from measurement", "Documented at the code that implements them");

  const rows = [
    ["Whitening the descriptors", "AUROC 0.711 → 0.944",
     "Distance was decided by whichever feature group had the biggest units — an accident of units, not a statement about defects."],
    ["Overlapping patches", "scratch recall 38% → 100%",
     "A thin defect crossing a tile boundary is diluted into two patches and caught in neither."],
    ["Two background kernels", "dent recall 88% → 100%",
     "A 40 px dent inside a 65 px background window is absorbed into its own background and vanishes."],
    ["A calibration bug", "escape rate 57.5% → 0%",
     "Leave-one-out folds built half-size coresets, inflating every distance and pushing thresholds too high."],
  ];
  let y = 1.85;
  rows.forEach(([k, delta, why]) => {
    card(s, M, y, W - 2 * M, 1.02);
    s.addText(k, { x: M + 0.3, y: y + 0.14, w: 3.5, h: 0.34, isTextBox: true, margin: 0,
      fontFace: H, fontSize: 15, bold: true, color: FG });
    s.addText(delta, { x: M + 0.3, y: y + 0.52, w: 3.5, h: 0.32, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13, bold: true, color: GREEN });
    s.addText(why, { x: M + 4.1, y: y + 0.2, w: W - 2 * M - 4.5, h: 0.7, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13, color: MUT, lineSpacing: 18 });
    y += 1.13;
  });

  s.addText("At the point the escape rate was 57.5%, AUROC was already 0.944. The detector was fine — the thresholds were broken.", {
    x: M, y: 6.42, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 13, italic: true, color: AMBER });
  footer(s, 8);
  s.addNotes("A separability metric cannot see a calibration bug. That is why the repo reports escapes and false rejects as first-class numbers.");
}

/* ── 9 · Why on-device ─────────────────────────────────────────────────── */
{
  const s = slide();
  title(s, "On-device is a requirement here, not a preference");

  const rows = [
    ["Latency", "200–500 ms + uplink", "~10 ms, deterministic"],
    ["No connectivity", "stops", "unaffected"],
    ["Part geometry (customer IP)", "leaves the building", "never leaves the device"],
    ["Marginal cost per part", "per-call API fee", "electricity"],
    ["Shift reports, 11 lines × 365 days", "~120,000 LLM calls/yr", "₹0"],
  ];

  s.addText("", { x: 0, y: 0, w: 0.1, h: 0.1, isTextBox: true });
  s.addText("Cloud", { x: 5.6, y: 1.78, w: 3.2, h: 0.32, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 14, bold: true, color: RED });
  s.addText("NIRIKSHAK", { x: 9.1, y: 1.78, w: 3.4, h: 0.32, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 14, bold: true, color: GREEN });

  let y = 2.25;
  rows.forEach(([k, a, b], i) => {
    if (i % 2 === 0) card(s, M - 0.15, y - 0.08, W - 2 * M + 0.3, 0.68, "1A1F24");
    s.addText(k, { x: M, y, w: 4.7, h: 0.4, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13.5, color: FG });
    s.addText(a, { x: 5.6, y, w: 3.3, h: 0.4, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13.5, color: MUT });
    s.addText(b, { x: 9.1, y, w: 3.4, h: 0.4, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13.5, bold: true, color: FG });
    y += 0.78;
  });

  s.addText("That last row is why per-shift quality analysis does not exist in these plants today. Nobody will approve the recurring bill.\nOn the NPU it runs every shift on every line, and nobody has to justify it.", {
    x: M, y: 6.15, w: W - 2 * M, h: 0.7, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 13.5, color: AMBER, lineSpacing: 19 });
  footer(s, 9);
}

/* ── 10 · Why the UNO Q ────────────────────────────────────────────────── */
{
  const s = slide();
  title(s, "Why an Arduino UNO Q is in this system", "Not decoration — it removes 97% of the work");

  card(s, M, 1.85, 6.0, 2.15);
  s.addText("The gate converts a video problem into a parts problem", {
    x: M + 0.3, y: 2.05, w: 5.4, h: 0.4, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 16, bold: true, color: FG });
  s.addText("A camera at 30 fps produces 108,000 frames/hour.\nA line at 40 parts/min produces 2,400 parts.\n\nWithout the gate, 97% of NPU work is spent on empty conveyor.", {
    x: M + 0.3, y: 2.5, w: 5.4, h: 1.35, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 13.5, color: MUT, lineSpacing: 19 });

  card(s, 7.0, 1.85, W - M - 7.0, 2.15);
  s.addText("Determinism belongs on the Cortex-M33", {
    x: 7.3, y: 2.05, w: 5.0, h: 0.4, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 16, bold: true, color: FG });
  s.addText("A reject arriving 30 ms late has diverted the wrong part.\n\n30 ms of scheduling jitter is an ordinary Tuesday for a Linux userspace process — so the diverter fires from the STM32U585 under Zephyr, not from Python.", {
    x: 7.3, y: 2.5, w: 5.0, h: 1.4, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 13, color: MUT, lineSpacing: 18 });

  card(s, M, 4.25, W - 2 * M, 1.55);
  s.addText("And the gate is deliberately not a neural network.", {
    x: M + 0.35, y: 4.45, w: 11.5, h: 0.4, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 16, bold: true, color: AMBER });
  s.addText("Frame differencing plus a sharpness check answers “is a part in frame?” correctly in ~2 ms. A model there would need its own enrolment, its own failure modes and its own explanation to the operator — to answer a question background subtraction already answers. Complexity has to earn its place.", {
    x: M + 0.35, y: 4.88, w: 11.5, h: 0.85, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 13, color: MUT, lineSpacing: 18 });
  footer(s, 10);
}

/* ── 11 · Deployment / cost ────────────────────────────────────────────── */
{
  const s = slide();
  title(s, "What it costs to put on a line");

  const bom = [
    ["HP Snapdragon PC (X-series)", "95,000"],
    ["Arduino UNO Q (4 GB)", "6,000"],
    ["USB camera, 1080p fixed focus", "3,000"],
    ["LED ring light + mount", "4,000"],
    ["Diverter solenoid + 24 V supply", "3,000"],
  ];
  s.addText("Bill of materials", { x: M, y: 1.8, w: 5.5, h: 0.32, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 11.5, color: MUT, charSpacing: 0.8 });
  let y = 2.22;
  bom.forEach(([k, v]) => {
    s.addText(k, { x: M, y, w: 4.3, h: 0.36, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13.5, color: FG });
    s.addText("₹ " + v, { x: 4.5, y, w: 1.5, h: 0.36, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13.5, color: MUT, align: "right" });
    y += 0.46;
  });
  s.addShape(pres.ShapeType.line, { x: M, y: y + 0.06, w: 5.3, h: 0,
    line: { color: "2A3138", width: 1 } });
  s.addText("Total", { x: M, y: y + 0.2, w: 4.3, h: 0.4, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 16, bold: true, color: FG });
  s.addText("₹ 1,11,000", { x: 3.4, y: y + 0.2, w: 2.6, h: 0.4, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 16, bold: true, color: GREEN, align: "right" });

  card(s, 6.7, 1.9, W - M - 6.7, 2.0);
  s.addText("Conventional machine-vision cell", { x: 7.0, y: 2.12, w: 5.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 12, color: MUT });
  s.addText("₹ 15–40 lakh", { x: 7.0, y: 2.45, w: 5.0, h: 0.6, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 32, bold: true, color: RED });
  s.addText("Software cost zero. Marginal inference cost is electricity.", {
    x: 7.0, y: 3.15, w: 5.0, h: 0.55, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 12.5, color: MUT, lineSpacing: 17 });

  card(s, 6.7, 4.05, W - M - 6.7, 1.9);
  s.addText("A recipe is one 320 KB file", { x: 7.0, y: 4.25, w: 5.0, h: 0.35, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 15, bold: true, color: AMBER });
  s.addText("An operator copies it to the next cell on a USB stick. No account, no sync, no licence server — a deliberate product decision for this market.", {
    x: 7.0, y: 4.65, w: 5.0, h: 1.1, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 12.5, color: MUT, lineSpacing: 17 });

  s.addText("Commissioning: fixture and light the part once, enrol 20 good samples, run one advisory shift with the diverter disabled, retune, then go live.", {
    x: M, y: 6.2, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 12.5, italic: true, color: "5A646E" });
  footer(s, 11);
}

/* ── 12 · Limits + roadmap ─────────────────────────────────────────────── */
{
  const s = slide();
  title(s, "Limits we know about, and what happens next");

  const limits = [
    "Synthetic validation only. MVTec AD and a real camera trial are the next milestone.",
    "NPU latencies are design targets from Qualcomm's published profiles. Our measured numbers are the CPU reference path.",
    "Registration assumed — a tumbling part on an unfixtured belt needs pos_weight = 0, costing sensitivity.",
    "Reflective and transparent parts are hard. Polarised lighting is the fix, and it is a cell-design problem.",
    "Not a metrology tool. It finds departures from normal; it does not measure a dimension to tolerance.",
  ];
  const next = [
    "Profile the real backbone on Snapdragon X2 Elite via AI Hub — replace every modelled number with a measured one.",
    "MVTec AD benchmark run; the evaluator already accepts its layout unchanged.",
    "Wire Tier 2 and Tier 3 to GenieX on ARM64 and measure the true escalation cost.",
    "Camera trial on a real line, in advisory mode, with a supervisor reviewing the REVIEW bin.",
  ];

  s.addText("Limitations", { x: M, y: 1.78, w: 5.6, h: 0.35, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 16, bold: true, color: RED });
  s.addText(limits.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < limits.length - 1 } })), {
    x: M, y: 2.25, w: 5.6, h: 3.3, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 13, color: FG, lineSpacing: 19, paraSpaceAfter: 9 });

  s.addText("Next round", { x: 7.1, y: 1.78, w: 5.5, h: 0.35, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 16, bold: true, color: GREEN });
  s.addText(next.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < next.length - 1 } })), {
    x: 7.1, y: 2.25, w: 5.5, h: 3.3, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 13, color: FG, lineSpacing: 19, paraSpaceAfter: 9 });

  card(s, M, 5.7, W - 2 * M, 1.0);
  s.addText("Every number in this deck is reproducible from the repository: two commands regenerate the data, the results and the benchmarks.", {
    x: M + 0.35, y: 5.88, w: W - 2 * M - 0.7, h: 0.4, isTextBox: true, margin: 0,
    fontFace: H, fontSize: 15, color: AMBER });
  s.addText("github.com/jashpasad/NIRIKSHAK      ·      Jash Pasad, IIT Gandhinagar", {
    x: M + 0.35, y: 6.3, w: W - 2 * M - 0.7, h: 0.32, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 13, color: FG });
  footer(s, 12);
  s.addNotes("Closing line: every claim is checkable rather than believable. That is the point of shipping the generator and the evaluator alongside the model.");
}

pres.writeFile({ fileName: path.join(__dirname, "NIRIKSHAK_pitch.pptx") })
  .then(f => console.log("wrote", f));
