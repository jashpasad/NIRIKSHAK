const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, LevelFormat,
} = require("docx");
const fs = require("fs");
const path = require("path");

const ACCENT = "B26A00";
const MUT = "5A646E";
const DARK = "1A1E22";

const P = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 140, line: o.line ?? 280 },
  alignment: o.align,
  children: [new TextRun({
    text, font: "Calibri", size: o.size ?? 21,
    bold: o.bold, italics: o.italics, color: o.color ?? "1A1E22",
  })],
});

const Rich = (runs, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 140, line: 280 },
  children: runs.map(r => new TextRun({
    text: r.t, font: "Calibri", size: r.size ?? 21,
    bold: r.b, italics: r.i, color: r.c ?? "1A1E22",
  })),
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 300, after: 150 },
  children: [new TextRun({ text, font: "Arial", size: 26, bold: true, color: DARK })],
});

const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 220, after: 110 },
  children: [new TextRun({ text, font: "Arial", size: 22, bold: true, color: ACCENT })],
});

const Bullet = (text) => new Paragraph({
  numbering: { reference: "bul", level: 0 },
  spacing: { after: 90, line: 280 },
  children: [new TextRun({ text, font: "Calibri", size: 21, color: DARK })],
});

const cell = (text, { b = false, w, c, shade, align } = {}) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
  margins: { top: 90, bottom: 90, left: 130, right: 130 },
  children: [new Paragraph({
    alignment: align,
    spacing: { after: 0, line: 260 },
    children: [new TextRun({ text, font: "Calibri", size: 19, bold: b, color: c ?? DARK })],
  })],
});

function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: "D6DBE0" },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: "D6DBE0" },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "E6EAEE" },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) =>
          cell(h, { b: true, w: widths[i], shade: "F2F4F6",
                    align: i ? AlignmentType.RIGHT : undefined })),
      }),
      ...rows.map(r => new TableRow({
        children: r.map((v, i) =>
          cell(v, { w: widths[i], b: i === 0 ? false : false,
                    align: i ? AlignmentType.RIGHT : undefined })),
      })),
    ],
  });
}

const W = [3400, 1500, 1500, 2600];

const children = [
  new Paragraph({
    spacing: { after: 40 },
    children: [new TextRun({ text: "NIRIKSHAK", font: "Arial", size: 46, bold: true, color: DARK })],
  }),
  Rich([{ t: "निरीक्षक", size: 22, c: ACCENT }, { t: "  ·  “the inspector”", size: 22, c: ACCENT }],
       { after: 100 }),
  P("Zero-shot visual quality inspection on Snapdragon", { size: 26, bold: true, after: 80 }),
  P("Jash Pasad", { size: 23, bold: true, after: 30 }),
  P("B.Tech · Indian Institute of Technology Gandhinagar", { size: 19, color: MUT, after: 30 }),
  P("Snapdragon® AI Lab Build & Present Challenge 2026  ·  Individual submission", { size: 19, color: MUT, after: 30 }),
  P("Repository:  github.com/jashpasad/NIRIKSHAK", { size: 19, color: ACCENT, after: 300 }),

  Rich([
    { t: "An HP Snapdragon PC and an Arduino UNO Q become a factory inspection cell that learns a new part from roughly twenty good samples. No labels, no training, no cloud. ", b: true },
    { t: "The system runs four models of increasing cost, arranged so that expensive ones only ever see the parts the cheap ones could not settle." },
  ], { after: 260 }),

  H1("1. The problem"),
  P("India has roughly 6.3 crore MSMEs. Almost none of them perform automated visual inspection, and the reason is not that they do not want to."),
  Bullet("Cost. A conventional machine-vision cell (Cognex, Keyence, Omron) costs ₹15–40 lakh per line — more than the annual profit of most units that would benefit from one."),
  Bullet("Data that does not exist. The standard AI answer is to train a detector on labelled defects. A workshop producing 400 parts a day at a 2% reject rate generates eight defective parts a day, unlabelled, across defect types nobody has enumerated. Collecting a training set takes years, and the resulting model still cannot detect a defect type it has never seen. In manufacturing, the defect you have never seen is the one that reaches the customer."),
  Bullet("Cloud does not fix it. A conveyor does not wait for a 300 ms round trip; shop floors in Rajkot, Ludhiana and Coimbatore lack reliable uplinks; and a job shop is typically barred by contract from uploading images of its customer's parts, because the part geometry is the customer's intellectual property."),
  P("The market is therefore stuck between “₹40 lakh” and “nothing”, and picks nothing.",
    { italics: true, color: ACCENT, after: 220 }),

  H1("2. The approach"),
  Rich([
    { t: "A defect is not a thing you learn to recognise. A defect is a region that does not look like anything you saw on a good part.", b: true, size: 23 },
  ], { after: 150 }),
  P("NIRIKSHAK models normality rather than defects. Every factory already possesses that training data — it is the parts they ship. Twenty of them is sufficient. This single inversion removes every constraint above at once: no labels are needed because good parts require none; no defect dataset is needed because defects are never modelled; unknown defect types are caught by construction, since anything unlike normal is anomalous; and enrolment is indexing rather than gradient descent, so it completes in about five seconds on a laptop."),

  H1("3. Technical implementation"),
  H2("3.1  The cascade: compute proportional to uncertainty"),
  P("A naive design runs a vision-language model on every part. On a Snapdragon X2 Elite that costs roughly 400 ms per part, capping the line at about 2.5 parts per second, pinning the NPU at full draw all shift, and producing paragraphs of prose about parts that were perfectly fine."),
  P("NIRIKSHAK instead spends compute in proportion to how uncertain it is:"),
  table(
    ["Tier", "Runs on", "Cost", "Fires on"],
    [
      ["0 · Gate", "Arduino UNO Q (QRB2210)", "~2 ms", "every camera frame"],
      ["1 · Screen", "Hexagon NPU", "~10 ms", "every part"],
      ["2 · Explain", "Hexagon NPU (Qwen3-VL-4B)", "~400 ms", "only the uncertainty band"],
      ["3 · Reason", "Hexagon NPU (Qwen3-4B)", "~25 s", "once per shift"],
    ], W),
  P("", { after: 120 }),
  P("At a realistic 97% first-pass yield, Tier 2 fires on about 4% of parts, so the mean cost per part is 10 + 0.04 × 400 ≈ 26 ms rather than 400 ms. That is a six-fold throughput gain obtained purely by ordering the models, without making any single model faster."),
  P("Two properties matter as much as the speed. First, the generative model never decides: Tier 1's calibrated distance determines PASS / REVIEW / FAIL, and Tier 2 supplies only vocabulary for the operator. Letting a language model adjudicate a physical accept/reject would be indefensible on an audited line. Second, the escalation policy is an explicit rule set rather than a model, because on an audited line the question “why was this part escalated?” must have an answer a person can read."),

  H2("3.2  Tier 1 — the screen"),
  P("A mid-level CNN backbone runs on the Hexagon NPU through the ONNX Runtime QNN Execution Provider, producing a grid of local patch descriptors. We concatenate an early feature map (texture, edges) with a deeper one (shape, part identity) and deliberately avoid the final layers: ImageNet's last block is optimised for “which of a thousand classes is this”, which is precisely the wrong invariance here — a scratch must not be invariant."),
  P("Descriptors are whitened per dimension against the enrolment set, then L2-normalised, which makes the nearest-neighbour search a single matrix product. The normality bank is compressed by greedy k-center coreset selection to about 5% of the enrolled descriptors. Coreset selection is chosen over random subsampling specifically because it retains the rare-but-legitimate patches — a stamped logo, a chamfer, a colour transition — which are exactly the good-part features whose loss causes false rejects."),
  P("Thresholds are calibrated by leave-one-out over the enrolment set: each image is scored against a bank that has never seen it. Two thresholds are produced, not one; between them lies the REVIEW band that Tier 2 exists to resolve."),

  H2("3.3  Tier 0 — why an Arduino UNO Q is in the system"),
  P("A camera at 30 fps produces 108,000 frames per hour, while a line at 40 parts per minute produces 2,400 parts. Screening frames rather than parts would waste 97% of the NPU's work on empty conveyor. The gate converts a video problem into a parts problem, and it is the single largest efficiency gain in the design."),
  P("It runs frame differencing and a sharpness check on the QRB2210's CPU — deliberately not a neural network, because a model there would require its own enrolment, its own failure modes and its own explanation to the operator, all to answer a question background subtraction answers correctly in two milliseconds."),
  P("The reject diverter fires from the STM32U585 under Zephyr rather than from Python on Linux. A reject arriving 30 ms late has diverted the wrong part, and 30 ms of scheduling jitter is unremarkable for a Linux userspace process. Perception belongs on the big cores; determinism belongs on the Cortex-M33 — which is precisely why the UNO Q has one."),

  H1("4. Results"),
  P("24 enrolment samples, 40 held-out good parts, 40 defective parts across five defect types. Every figure below is reproducible from the repository with two commands."),
  table(
    ["Defect type", "n", "AUROC", "caught / escaped"],
    [
      ["scratch", "8", "1.000", "8 / 0"],
      ["dent", "8", "1.000", "8 / 0"],
      ["chip", "8", "1.000", "8 / 0"],
      ["contamination", "8", "1.000", "8 / 0"],
      ["discolouration", "8", "1.000", "8 / 0"],
      ["Overall", "40", "1.000", "40 / 0"],
    ], W),
  P("", { after: 120 }),
  P("Escape rate 0%. False reject rate 0 of 40. All 40 defects localised. Enrolment 5.3 s. The evaluator additionally reports the threshold safety window — the range of sigma values giving simultaneously zero escapes and zero false rejects — which on this dataset is [0.78, 2.06]; we ship 1.50, the centre, rather than a value tuned to the edge."),

  H2("An honest caveat about that 1.000"),
  Rich([{ t: "We generated that data, so we do not get to be impressed by it. ", b: true },
        { t: "It demonstrates that the pipeline is correct and reproducible on any machine; it is not a claim about field accuracy." }], { after: 130 }),
  P("What makes it non-trivial rather than meaningless is that the generator injects the nuisance variation which normally defeats this class of method: a randomised lighting gradient, ±4 px translation, ±1.4° rotation, Gaussian sensor noise, and a brushed-metal finish whose parallel streaks are visually confusable with scratches. Without those, any detector separates a defect from a pixel-identical template and the benchmark is a lie."),
  P("The external reference that carries weight is MVTec AD, the standard public benchmark, where published PatchCore-class methods reach approximately 0.99 image AUROC. Our evaluation harness accepts MVTec's directory layout unchanged; running it, alongside a real camera trial, is the first task of the next round."),

  H2("Cascade economics"),
  table(
    ["Line yield", "Escalation", "ms / part", "vs VLM on every part"],
    [
      ["90%", "10.9%", "95.6", "4.2×"],
      ["95%", "6.0%", "75.8", "5.3×"],
      ["97%", "4.0%", "67.9", "5.9×"],
      ["99%", "2.0%", "60.0", "6.7×"],
    ], W),
  P("", { after: 120 }),
  P("The gating gain grows as the line improves, which is the correct incentive: a well-run line pays almost nothing for the explanation tier. The Tier-1 figure is measured on the pessimistic path — the CPU reference descriptor, with no NPU and no CNN backbone."),

  H1("5. Engineering findings"),
  P("Four changes moved the numbers more than everything else combined. Each was found by measurement and is documented at the code that implements it."),
  Bullet("Whitening the descriptors: AUROC 0.711 → 0.944. Raw descriptors mix quantities with very different natural scales, so Euclidean distance was decided by whichever feature group had the largest units — an accident of units rather than a statement about defects."),
  Bullet("Overlapping patches: scratch recall 38% → 100%. With non-overlapping tiles, a thin defect crossing a tile boundary is diluted into two patches and detected in neither."),
  Bullet("Two background kernels: dent recall 88% → 100%. A single local-background width can only see defects smaller than itself; a 40 px dent inside a 65 px background window is absorbed into its own background and disappears."),
  Bullet("A calibration bug: escape rate 57.5% → 0%. Leave-one-out folds were building half-size coresets, inflating every fold distance and placing thresholds too high. At that point AUROC was already 0.944 — the detector was fine and the thresholds were broken, which is exactly the failure a separability metric cannot see. This is why the repository reports escapes and false rejects as first-class numbers."),
  P("Separately, replacing the per-fold coreset rebuild with origin masking reduced enrolment from 92.7 s to 5.3 s. Ninety seconds is a long time to stand at a machine holding a part."),

  H1("6. Why on-device is a requirement, not a preference"),
  table(
    ["", "Cloud", "NIRIKSHAK", ""],
    [
      ["Latency", "200–500 ms", "~10 ms", "deterministic"],
      ["No connectivity", "stops", "unaffected", "no network dependency"],
      ["Part geometry", "leaves the building", "never leaves", "customer IP"],
      ["Cost per part", "API fee", "electricity", "zero marginal"],
      ["Shift reports / year", "~120,000 calls", "₹0", "11 lines × 365 days"],
    ], W),
  P("", { after: 120 }),
  P("The last row explains why per-shift quality analysis does not exist in these plants today. It is not that nobody wants it — it is that nobody will approve the recurring bill. On the NPU it runs every shift on every line and nobody has to justify it."),

  H1("7. Deployment"),
  P("Bill of materials: HP Snapdragon PC ~₹95,000; Arduino UNO Q (4 GB) ~₹6,000; USB camera ~₹3,000; LED ring light and mount ~₹4,000; diverter solenoid and 24 V supply ~₹3,000. Total approximately ₹1,11,000, against ₹15–40 lakh for a conventional machine-vision cell. Software cost is zero and marginal inference cost is electricity."),
  P("A part recipe is a single 320 KB file that an operator copies to the next cell on a USB stick — no account, no synchronisation, no licence server. That is a deliberate product decision for this market rather than an oversight."),
  P("Commissioning: fixture and light the part once, enrol twenty good samples, run one shift in advisory mode with the diverter disabled while the supervisor reviews the REVIEW bin, retune the threshold against that shift's data, then go live. The advisory shift is not optional; shipping thresholds derived from twenty images straight into a live reject loop is how an operator loses trust in the machine in a single morning, and an operator who switches it off has a system that detects nothing."),

  H1("8. Limitations"),
  Bullet("Validation is synthetic so far. MVTec AD and a real camera trial are the next milestone."),
  Bullet("The NPU latency figures are design targets taken from Qualcomm's published Snapdragon X2 Elite profiles. Our measured numbers are from the CPU reference path. Replacing every modelled figure with a profiled one is the first task of the next round."),
  Bullet("The reference descriptor is a floor, not the product — it exists so the repository runs with zero downloads. The production path is the NPU backbone, and the two are benchmarked separately and never mixed."),
  Bullet("Registration is assumed. A tumbling part on an unfixtured belt requires the positional feature to be disabled, at some cost in sensitivity."),
  Bullet("Reflective and transparent parts are hard; specular highlights move with the part and read as anomalies. Polarised lighting is the standard fix and is a cell-design problem rather than a software one."),
  Bullet("This is not a metrology tool. It finds departures from normal; it does not measure a dimension to tolerance."),

  H1("9. Repository"),
  P("The complete implementation, the synthetic data generator, the evaluation harness and every number quoted above are in the repository under Apache-2.0, so that each claim can be checked rather than believed. Eleven behavioural tests cover the properties that would otherwise fail silently — a system that still runs and still prints verdicts while being wrong is the only failure mode that matters on a production line."),
  P("Two commands regenerate the data, the results and the benchmarks:", { after: 80 }),
  new Paragraph({
    spacing: { after: 60 },
    children: [new TextRun({ text: "python scripts/make_demo_data.py", font: "Courier New", size: 19, color: DARK })],
  }),
  new Paragraph({
    spacing: { after: 220 },
    children: [new TextRun({ text: "python scripts/demo.py", font: "Courier New", size: 19, color: DARK })],
  }),
  P("github.com/jashpasad/NIRIKSHAK", { size: 21, bold: true, color: ACCENT, after: 40 }),
  P("Jash Pasad · Indian Institute of Technology Gandhinagar · Apache-2.0",
    { size: 19, color: MUT, after: 240 }),
];

const doc = new Document({
  numbering: {
    config: [{
      reference: "bul",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 340, hanging: 200 } } },
      }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },   // US Letter
        margin: { top: 1080, bottom: 1080, left: 1180, right: 1180 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, "NIRIKSHAK_description.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
