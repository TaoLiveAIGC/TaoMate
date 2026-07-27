const MANIFEST_PATH = "./assets/cases-manifest.json";
const CASE_GROUPS = [
  {
    key: "feature",
    selector: "#feature-video [data-case-track]",
    fallbackText: "Thirty-minute uninterrupted generation preview.",
  },
  {
    key: "long",
    selector: "#long-video [data-case-track]",
    fallbackText: "Minute-level generation preview.",
  },
  {
    key: "short",
    selector: "#short-video [data-case-track]",
    fallbackText: "Short clip generation preview.",
  },
];
const carouselStates = new WeakMap();
const CASE_PREVIEW_VOLUME = 0.86;
const GALLERY_SECTION_IDS = new Set(["feature-video", "long-video", "short-video"]);

function normalizeNavHrefForSection(sectionId) {
  return GALLERY_SECTION_IDS.has(sectionId) ? "#feature-video" : `#${sectionId}`;
}

function pauseCaseVideos(exceptVideo = null) {
  document.querySelectorAll(".case-card video").forEach((video) => {
    if (video === exceptVideo) {
      return;
    }

    video.pause();
    video.muted = true;
  });
}

function ensureCaseVideoLoaded(video) {
  if (video.dataset.loaded === "true") {
    return true;
  }

  const src = video.dataset.src;
  if (!src) {
    return false;
  }

  const source = document.createElement("source");
  source.src = src;
  source.type = video.dataset.type || "video/mp4";
  video.append(source);
  video.preload = "auto";
  video.load();
  video.dataset.loaded = "true";
  return true;
}

function syncActiveCaseVideo(carousel, cards, activeIndex) {
  const activeCard = cards[activeIndex];

  cards.forEach((card, cardIndex) => {
    const video = card.querySelector("video");
    if (!video) {
      return;
    }

    if (cardIndex !== activeIndex) {
      video.pause();
      video.muted = true;
      return;
    }
  });

  if (carousel.dataset.videoLoadingEnabled !== "true") {
    return;
  }

  const activeVideo = activeCard?.querySelector("video");
  if (activeVideo) {
    ensureCaseVideoLoaded(activeVideo);
  }
}

function enableCarouselVideoLoading(carousel) {
  if (carousel.dataset.videoLoadingEnabled === "true") {
    return;
  }

  carousel.dataset.videoLoadingEnabled = "true";
  const state = carouselStates.get(carousel) || { index: 0 };
  syncActiveCaseVideo(carousel, getCarouselCards(carousel), state.index);
}

function enableVisibleCaseLoading() {
  const carousels = Array.from(document.querySelectorAll("[data-case-carousel]")).filter(
    (carousel) => !carousel.classList.contains("is-empty") && getCarouselCards(carousel).length,
  );

  if (!carousels.length) {
    return;
  }

  enableCarouselVideoLoading(carousels[0]);

  if (!("IntersectionObserver" in window)) {
    carousels.slice(1).forEach((carousel) => enableCarouselVideoLoading(carousel));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        enableCarouselVideoLoading(entry.target);
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "180px 0px", threshold: 0.12 },
  );

  carousels.slice(1).forEach((carousel) => observer.observe(carousel));
}

function playCaseVideo(video, { restart = false, audible = true } = {}) {
  if (!ensureCaseVideoLoaded(video)) {
    return;
  }

  pauseCaseVideos(video);

  if (restart) {
    video.currentTime = 0;
  }

  video.volume = CASE_PREVIEW_VOLUME;
  video.muted = !audible;

  video.play().catch(() => {
    video.muted = true;
    video.play().catch(() => {});
  });
}

function updateCaseProgress(video) {
  if (video.dataset.scrubbing === "true") {
    return;
  }

  const media = video.closest(".case-media");
  const progress = media?.querySelector("[data-case-progress]");

  if (!progress) {
    return;
  }

  const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
  const percent = duration ? (video.currentTime / duration) * 100 : 0;
  progress.value = String(Math.min(100, Math.max(0, percent)));
  progress.style.setProperty("--progress", `${percent.toFixed(2)}%`);
}

function seekCaseVideo(video, progress) {
  if (!ensureCaseVideoLoaded(video)) {
    return;
  }

  const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
  if (!duration) {
    return;
  }

  const percent = Math.min(100, Math.max(0, Number.parseFloat(progress.value) || 0));
  video.currentTime = (percent / 100) * duration;
  progress.style.setProperty("--progress", `${percent}%`);
  updateCaseProgress(video);
}

function disableCaseExpansion() {
  document.querySelectorAll(".case-media").forEach((media) => {
    media.removeAttribute("aria-disabled");
    media.removeAttribute("tabindex");
    media.removeAttribute("data-video");
    media.removeAttribute("data-prompt");
  });
}

document.addEventListener(
  "click",
  (event) => {
    if (event.target.closest("[data-case-control]")) {
      return;
    }

    const media = event.target.closest(".case-media");
    if (!media) {
      return;
    }

    event.preventDefault();
    event.stopImmediatePropagation();

    const video = media.querySelector("video");
    if (!video) {
      return;
    }

    if (video.paused) {
      playCaseVideo(video, { audible: true });
      return;
    }

    video.pause();
    video.muted = true;
  },
  true,
);

document.addEventListener("keydown", (event) => {
  if (event.target.closest("[data-case-control]")) {
    return;
  }

  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }

  const media = event.target.closest(".case-media");
  if (!media) {
    return;
  }

  event.preventDefault();
  media.click();
});

function getCarouselCards(carousel) {
  return Array.from(carousel.querySelectorAll(".case-card"));
}

function replaceCarouselButton(button) {
  const clone = button.cloneNode(true);
  button.replaceWith(clone);
  return clone;
}

function prepareSpotlightCarousel(carousel) {
  if (carousel.dataset.spotlightReady === "true") {
    return;
  }

  const prev = carousel.querySelector("[data-carousel-prev]");
  const next = carousel.querySelector("[data-carousel-next]");

  if (!prev || !next) {
    return;
  }

  const spotlightPrev = replaceCarouselButton(prev);
  const spotlightNext = replaceCarouselButton(next);
  const navigate = (event, direction) => {
    event.preventDefault();
    event.stopImmediatePropagation();
    pauseCaseVideos();

    const state = carouselStates.get(carousel) || { index: 0 };
    renderSpotlightCarousel(carousel, state.index + direction);
  };

  spotlightPrev.addEventListener("click", (event) => navigate(event, -1));
  spotlightNext.addEventListener("click", (event) => navigate(event, 1));
  carousel.dataset.spotlightReady = "true";
}

function renderSpotlightCarousel(carousel, requestedIndex = 0) {
  const viewport = carousel.querySelector(".case-carousel-viewport");
  const track = carousel.querySelector("[data-case-track]");
  const prev = carousel.querySelector("[data-carousel-prev]");
  const next = carousel.querySelector("[data-carousel-next]");
  const status = carousel.querySelector("[data-carousel-status]");
  const cards = getCarouselCards(carousel);

  if (!viewport || !track || !prev || !next) {
    return;
  }

  carousel.classList.add("gallery-spotlight");
  carousel.classList.toggle("is-empty", cards.length === 0);
  carousel.classList.toggle("is-single", cards.length <= 1);

  if (!cards.length) {
    track.style.transform = "translate3d(0, 0, 0)";
    if (status) {
      status.textContent = "0 / 0";
    }
    prev.disabled = true;
    next.disabled = true;
    carouselStates.set(carousel, { index: 0 });
    return;
  }

  const index = ((requestedIndex % cards.length) + cards.length) % cards.length;
  const activeCard = cards[index];
  const centerOffset = (viewport.clientWidth - activeCard.offsetWidth) / 2;
  const translateX = centerOffset - activeCard.offsetLeft;

  track.style.transform = `translate3d(${translateX.toFixed(2)}px, 0, 0)`;
  if (status) {
    status.textContent = `${index + 1} / ${cards.length}`;
  }
  prev.disabled = cards.length <= 1;
  next.disabled = cards.length <= 1;

  cards.forEach((card, cardIndex) => {
    card.classList.toggle("is-active", cardIndex === index);
    card.classList.toggle("is-neighbor", Math.abs(cardIndex - index) === 1);
  });
  syncActiveCaseVideo(carousel, cards, index);

  carouselStates.set(carousel, { index });
}

function renderAllSpotlightCarousels() {
  document.querySelectorAll("[data-case-carousel]").forEach((carousel) => {
    prepareSpotlightCarousel(carousel);
    const state = carouselStates.get(carousel) || { index: 0 };
    renderSpotlightCarousel(carousel, state.index);
  });
}

// Hovering a card starts (or resumes) playback. Leaving the card keeps the
// video playing; playback only stops when the visitor clicks the video again,
// starts another case, or navigates the carousel.
document.addEventListener("pointerover", (event) => {
  const card = event.target.closest(".case-card");
  if (!card || card.contains(event.relatedTarget)) {
    return;
  }

  const video = card.querySelector("video");
  if (!video) {
    return;
  }

  playCaseVideo(video, { audible: true });
});

function syncMergedGalleryNav() {
  const navLinks = Array.from(document.querySelectorAll(".nav-links a"));
  const observedSections = Array.from(document.querySelectorAll("main section[id]"));

  if (!navLinks.length || !observedSections.length) {
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        const activeHref = normalizeNavHrefForSection(entry.target.id);
        navLinks.forEach((link) => {
          link.classList.toggle("active", link.getAttribute("href") === activeHref);
        });
      });
    },
    { threshold: 0.34 },
  );

  observedSections.forEach((section) => observer.observe(section));
}

function createCaseCard(item, group) {
  const card = document.createElement("article");
  card.className = `case-card ${group.key === "short" ? "" : "long-case-card"}`.trim();
  card.dataset.category = group.key;

  const media = document.createElement("div");
  media.className = "case-media";
  media.role = "button";
  media.tabIndex = 0;
  media.dataset.title = item.title;
  media.setAttribute("aria-label", `${item.title} audio-video preview`);

  const video = document.createElement("video");
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = "none";
  video.dataset.src = item.src;
  video.dataset.type = item.type || "video/mp4";
  if (item.poster) {
    video.poster = item.poster;
  }

  const chip = document.createElement("span");
  chip.className = "play-chip";
  chip.textContent = "Play";

  const progress = document.createElement("input");
  progress.className = "case-progress";
  progress.type = "range";
  progress.min = "0";
  progress.max = "100";
  progress.step = "0.1";
  progress.value = "0";
  progress.dataset.caseProgress = "";
  progress.dataset.caseControl = "";
  progress.setAttribute("aria-label", `${item.title} playback progress`);
  progress.style.setProperty("--progress", "0%");

  video.addEventListener("timeupdate", () => updateCaseProgress(video));
  video.addEventListener("loadedmetadata", () => updateCaseProgress(video));
  video.addEventListener("durationchange", () => updateCaseProgress(video));

  const endScrub = () => {
    if (video.dataset.scrubbing !== "true") {
      return;
    }
    video.dataset.scrubbing = "false";
    if (video.dataset.resumeAfterScrub === "true") {
      video.dataset.resumeAfterScrub = "false";
      video.play().catch(() => {});
    }
  };

  progress.addEventListener("pointerdown", (event) => {
    event.stopPropagation();
    video.dataset.scrubbing = "true";
    if (!video.paused) {
      video.dataset.resumeAfterScrub = "true";
      video.pause();
    }
  });
  progress.addEventListener("pointerup", endScrub);
  progress.addEventListener("pointercancel", endScrub);
  progress.addEventListener("change", endScrub);
  progress.addEventListener("click", (event) => {
    event.stopPropagation();
  });
  progress.addEventListener("input", () => seekCaseVideo(video, progress));

  media.append(video, chip, progress);

  card.append(media);

  return card;
}

async function loadManifest() {
  if (window.ECHO_CASE_MANIFEST?.groups) {
    return window.ECHO_CASE_MANIFEST;
  }

  try {
    const response = await fetch(`${MANIFEST_PATH}?v=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load ${MANIFEST_PATH}: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(error);
    return { schemaVersion: 1, groups: {} };
  }
}

function renderManifestGroup(manifest, group) {
  const track = document.querySelector(group.selector);

  if (!track) {
    return false;
  }

  const items = Array.isArray(manifest?.groups?.[group.key]) ? manifest.groups[group.key] : [];
  track.replaceChildren(...items.map((item) => createCaseCard(item, group)));
  track.closest("[data-case-carousel]")?.classList.toggle("is-empty", items.length === 0);
  track.closest("section")?.classList.remove("is-empty-section");
  return true;
}

function renderManifestCases(manifest) {
  let renderedAny = false;
  for (const group of CASE_GROUPS) {
    renderedAny = renderManifestGroup(manifest, group) || renderedAny;
  }
  return renderedAny;
}

document.addEventListener("DOMContentLoaded", async () => {
  renderManifestCases({});
  await renderManifestCases(await loadManifest());
  disableCaseExpansion();
  syncMergedGalleryNav();

  requestAnimationFrame(() => {
    window.dispatchEvent(new Event("resize"));
    requestAnimationFrame(() => {
      renderAllSpotlightCarousels();
      enableVisibleCaseLoading();
    });
  });

  window.addEventListener("resize", () => {
    requestAnimationFrame(() => {
      renderAllSpotlightCarousels();
    });
  });
});
