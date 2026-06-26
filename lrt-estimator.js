(function (global) {
  const MODEL_URL = "lrt_model_4rank.json";
  const MODEL_VERSION = "4rank-2026-06-01";
  let modelPromise = null;

  function toFiniteNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function normalize(values) {
    const sum = values.reduce((total, value) => total + value, 0);
    return sum > 0 ? values.map((value) => value / sum) : values.map(() => 0);
  }

  function estimateWithModel(model, responses) {
    const priors = model.priors && model.priors.length === model.rankCount
      ? model.priors
      : Array.from({ length: model.rankCount }, () => 1 / model.rankCount);
    const logScores = priors.map((prior) => Math.log(Math.max(prior, 1e-12)));

    for (let itemIndex = 1; itemIndex <= model.itemCount; itemIndex += 1) {
      const itemName = `OOOs${itemIndex}`;
      const responseKey = `q${itemIndex}`;
      const category = toFiniteNumber(responses[responseKey]);
      const probabilities = model.itemProbabilityByRank[itemName]?.[category];

      if (!probabilities) {
        continue;
      }

      for (let rankIndex = 0; rankIndex < model.rankCount; rankIndex += 1) {
        logScores[rankIndex] += Math.log(Math.max(probabilities[rankIndex], 1e-12));
      }
    }

    const maxLogScore = Math.max(...logScores);
    const weights = logScores.map((score) => Math.exp(score - maxLogScore));
    const membership = normalize(weights);
    const maxMembership = Math.max(...membership);
    const rank = membership.indexOf(maxMembership) + 1;

    return {
      model_version: MODEL_VERSION,
      latent_rank: rank,
      rank_membership: membership,
      rank_membership_rank1: membership[0],
      rank_membership_rank2: membership[1],
      rank_membership_rank3: membership[2],
      rank_membership_rank4: membership[3],
      rank_membership_profile: membership.map((value) => Number(value.toFixed(6))).join(",")
    };
  }

  async function loadModel() {
    if (!modelPromise) {
      modelPromise = fetch(MODEL_URL, { cache: "no-store" }).then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load LRT model: ${response.status}`);
        }
        return response.json();
      });
    }

    return modelPromise;
  }

  async function estimate(responses) {
    const model = await loadModel();
    return estimateWithModel(model, responses);
  }

  global.OOOS_LRT = {
    modelVersion: MODEL_VERSION,
    loadModel,
    estimate,
    estimateWithModel
  };
})(window);
