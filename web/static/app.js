const form = document.getElementById("form");
const submitBtn = document.getElementById("submit");
const results = document.getElementById("results");
const list = document.getElementById("results-list");
const errorEl = document.getElementById("error");

const intFields = new Set([
  "pregnancies", "glucose", "blood_pressure",
  "skin_thickness", "insulin", "age"
]);

const PRESETS = {
  healthy: {
    pregnancies: 1, glucose: 85, blood_pressure: 66, skin_thickness: 29,
    insulin: 94, bmi: 26.6, diabetes_pedigree_function: 0.351, age: 31,
  },
  diabetic: {
    pregnancies: 8, glucose: 183, blood_pressure: 64, skin_thickness: 35,
    insulin: 168, bmi: 33.6, diabetes_pedigree_function: 0.627, age: 50,
  },
};

function applyPreset(preset) {
  for (const [name, value] of Object.entries(preset)) {
    const input = form.querySelector(`[name="${name}"]`);
    if (input) {
      input.value = value;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }
}

document.getElementById("preset-healthy")
  .addEventListener("click", (e) => { e.preventDefault(); applyPreset(PRESETS.healthy); });
document.getElementById("preset-diabetic")
  .addEventListener("click", (e) => { e.preventDefault(); applyPreset(PRESETS.diabetic); });

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting…";

  const data = {};
  for (const [key, value] of new FormData(form).entries()) {
    data[key] = intFields.has(key) ? parseInt(value, 10) : parseFloat(value);
  }

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail ? JSON.stringify(body.detail) : `Request failed (${res.status})`);
    }
    const json = await res.json();
    render(json.predictions);
    results.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict";
  }
});

function render(predictions) {
  list.innerHTML = "";
  for (const p of predictions) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = p.model;
    const tag = document.createElement("span");
    tag.className = "tag " + (p.prediction === 1 ? "positive" : "negative");
    tag.textContent = p.prediction === 1 ? "Positive" : "Negative";
    li.append(name, tag);
    list.appendChild(li);
  }
  results.hidden = false;
}
