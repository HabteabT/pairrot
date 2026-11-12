const API = "";
let currentSplit = "train";

function stageCard(stage) {
  const card = document.createElement("article");
  card.className = `card ${stage.state}`;
  const title = document.createElement("h2");
  title.textContent = stage.title;
  const pill = document.createElement("p");
  pill.className = "pill";
  pill.textContent = stage.state;
  const detail = document.createElement("p");
  detail.textContent = stage.detail;
  card.append(title, pill, detail);
  return card;
}

function renderStages(stages) {
  const board = document.querySelector("#stages");
  board.replaceChildren(...stages.map(stageCard));
}

function renderDataset(dataset) {
  const summary = document.querySelector("#dataset-summary");
  const words = document.querySelector("#words");
  const pairs = document.querySelector("#pairs");
  if (!dataset.ready) {
    summary.textContent = dataset.message;
    summary.className = "error";
    words.replaceChildren();
    pairs.replaceChildren();
    return;
  }

  const counts = dataset.label_counts[currentSplit];
  const empty = dataset.empty_documents_after_tokenization[currentSplit];
  summary.className = "";
  summary.textContent =
    `${currentSplit}: ${dataset.documents_per_split[currentSplit]} posts, ` +
    `${dataset.pairs_per_split[currentSplit]} pairs ` +
    `(${counts.y1} same category, ${counts.y0} different). ` +
    `${empty} posts are empty after cleaning. ` +
    `Vocabulary ${dataset.vocabulary_size}, feature width ${dataset.feature_width}.`;

  words.replaceChildren();
  for (const item of dataset.top_words) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${item.word} ${item.document_frequency}`;
    words.appendChild(chip);
  }

  const table = document.createElement("table");
  const head = document.createElement("tr");
  for (const label of ["Pair", "Y", "Post A", "Post B"]) {
    const cell = document.createElement("th");
    cell.textContent = label;
    head.appendChild(cell);
  }
  table.appendChild(head);
  for (const pair of dataset.pairs) {
    const row = document.createElement("tr");
    row.appendChild(textCell(pair.pair_id));
    const target = textCell(pair.y === "1" ? "1 same category" : "0 different categories");
    target.className = pair.y === "1" ? "same" : "different";
    row.appendChild(target);
    row.appendChild(textCell(`${pair.category_a}\n${pair.doc_a_id}`));
    row.appendChild(textCell(`${pair.category_b}\n${pair.doc_b_id}`));
    table.appendChild(row);
  }
  pairs.replaceChildren(table);
}

function textCell(value) {
  const cell = document.createElement("td");
  cell.textContent = value;
  return cell;
}

function markSplit() {
  for (const button of document.querySelectorAll("#splits button")) {
    button.classList.toggle("active", button.dataset.split === currentSplit);
  }
}

// Figures are written by the scripts in code/. Until they exist, show a line
// that says how to make them instead of a broken-image icon.
const FIGURES = [
  { id: "knows-when", src: "/knows_when.png", command: "python code/knows_when.py" },
  { id: "state-histogram", src: "/state_histogram.png", command: "train.bat" },
];

function refreshFigures() {
  for (const figure of FIGURES) {
    const image = document.querySelector(`#${figure.id}`);
    let note = image.nextElementSibling;
    if (!note || !note.classList.contains("missing")) {
      note = document.createElement("p");
      note.className = "missing";
      note.textContent = `This figure has not been drawn yet. Run ${figure.command}, then refresh.`;
      note.hidden = true;
      image.after(note);
    }
    image.onload = () => {
      image.hidden = false;
      note.hidden = true;
    };
    image.onerror = () => {
      image.hidden = true;
      note.hidden = false;
    };
    // The time stamp makes the browser ask again after a new training run.
    image.src = `${figure.src}?t=${Date.now()}`;
  }
}

let latestRequest = 0;

async function load() {
  const updated = document.querySelector("#updated");
  // Only the newest request may draw. A slow answer for a split the reader
  // has already left must not overwrite the page.
  const request = ++latestRequest;
  const split = currentSplit;
  try {
    const response = await fetch(`${API}/api/dashboard?split=${split}&limit=6`);
    const payload = await response.json();
    if (request !== latestRequest) {
      return;
    }
    if (!response.ok) {
      updated.textContent = `The backend reported a problem: ${payload.error || response.status}`;
      updated.className = "error";
      return;
    }
    updated.className = "";
    updated.textContent = `Updated ${payload.updated_at.replace("T", " ")}.`;
    renderStages(payload.stages);
    document.querySelector("#run-output").textContent = payload.run_output;
    renderDataset(payload.dataset);
  } catch (error) {
    if (request !== latestRequest) {
      return;
    }
    updated.textContent = "The dashboard is not running. Double-click start.bat in the Pairrot folder.";
    updated.className = "error";
    document.querySelector("#stages").replaceChildren();
  }
}

document.querySelector("#refresh").addEventListener("click", () => {
  refreshFigures();
  load();
});
for (const button of document.querySelectorAll("#splits button")) {
  button.addEventListener("click", () => {
    currentSplit = button.dataset.split;
    markSplit();
    load();
  });
}

markSplit();
refreshFigures();
load();
setInterval(load, 8000);
