// JoyAI-Echo page behaviors: intro scroll stage, typewriter, bibtex copy,
// fade-in reveals, nav active state, video preview modal.
document.addEventListener("DOMContentLoaded", () => {
    const root = document.documentElement;
    root.dataset.theme = "light";
    window.localStorage.removeItem("echo-theme");

    // ---- Intro hero stage (scroll-driven) ----
    const stage = document.querySelector("[data-intro-stage]");
    if (stage) {
        const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
        const easeOut = (t) => 1 - Math.pow(1 - t, 3);
        let ticking = null;

        const update = () => {
            const scrollable = Math.max(stage.offsetHeight - window.innerHeight, 1);
            const progress = clamp((window.scrollY - stage.offsetTop) / scrollable, 0, 1);
            const coverOut = easeOut(clamp((progress - 0.16) / 0.62, 0, 1));
            const nextIn = easeOut(clamp((progress - 0.34) / 0.48, 0, 1));
            const tintIn = easeOut(clamp((progress - 0.16) / 0.72, 0, 1));
            const coverOpacity = clamp(1 - coverOut * 1.18, 0, 1);

            document.body.classList.toggle("is-past-intro", progress > 0.84);
            const tintMax = root.dataset.theme === "light" ? 1 : 0.902;
            document.body.style.setProperty("--global-tint-opacity", (tintIn * tintMax).toFixed(3));

            if (reduceMotion) {
                const pastHalf = progress > 0.5;
                stage.style.setProperty("--intro-cover-opacity", pastHalf ? "0" : "1");
                stage.style.setProperty("--intro-next-opacity", pastHalf ? "1" : "0");
                stage.style.setProperty("--intro-next-transform", "translate3d(0, 0, 0)");
                return;
            }

            stage.style.setProperty("--intro-cover-opacity", coverOpacity.toFixed(3));
            stage.style.setProperty("--intro-echo-transform", `translate3d(${(-58 * coverOut).toFixed(2)}vw, 0, 0)`);
            stage.style.setProperty("--intro-long-transform", `translate3d(${(-58 * coverOut).toFixed(2)}vw, 0, 0)`);
            stage.style.setProperty("--intro-keywords-transform", `translate3d(0, ${(24 * coverOut).toFixed(2)}px, 0)`);
            stage.style.setProperty("--intro-keywords-opacity", clamp(1 - coverOut * 1.12, 0, 1).toFixed(3));
            stage.style.setProperty("--intro-scroll-transform", `translate3d(calc(-50% + ${(24 * coverOut).toFixed(2)}vw), ${(34 * coverOut).toFixed(2)}vh, 0) rotate(${(14 * coverOut).toFixed(2)}deg)`);
            stage.style.setProperty("--intro-scroll-opacity", clamp(1 - coverOut * 1.25, 0, 1).toFixed(3));
            stage.style.setProperty("--intro-next-opacity", nextIn.toFixed(3));
            stage.style.setProperty("--intro-next-transform", `translate3d(0, ${((1 - nextIn) * 46).toFixed(2)}px, 0)`);
        };

        const requestUpdate = () => {
            if (!ticking) {
                ticking = window.requestAnimationFrame(() => {
                    ticking = null;
                    update();
                });
            }
        };

        document.body.classList.add("intro-ready");
        update();
        window.addEventListener("scroll", requestUpdate, { passive: true });
        window.addEventListener("resize", update);
        document.addEventListener("echo-theme-change", requestUpdate);
    }

    // ---- Typewriter ----
    const typewriter = document.querySelector("[data-typewriter]");
    if (typewriter) {
        const phrases = (typewriter.dataset.phrases || "")
            .split("|")
            .map((phrase) => phrase.trim())
            .filter(Boolean);
        const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

        if (phrases.length && reduceMotion) {
            typewriter.textContent = phrases[0];
        } else if (phrases.length) {
            let phraseIndex = 0;
            let charIndex = 0;
            let deleting = false;
            const tick = () => {
                const phrase = phrases[phraseIndex];
                charIndex += deleting ? -1 : 1;
                typewriter.textContent = phrase.slice(0, charIndex);
                let delay = deleting ? 26 : 42;
                if (!deleting && charIndex === phrase.length) {
                    delay = 1500;
                    deleting = true;
                } else if (deleting && charIndex === 0) {
                    delay = 280;
                    deleting = false;
                    phraseIndex = (phraseIndex + 1) % phrases.length;
                }
                window.setTimeout(tick, delay);
            };
            tick();
        }
    }

    // ---- BibTeX copy ----
    const copyButton = document.querySelector("[data-copy-bibtex]");
    if (copyButton) {
        const code = copyButton.closest(".bibtex-card")?.querySelector("code");
        copyButton.addEventListener("click", async () => {
            if (!code) return;
            const text = code.textContent || "";
            try {
                await navigator.clipboard.writeText(text);
            } catch {
                const textarea = document.createElement("textarea");
                textarea.value = text;
                textarea.setAttribute("readonly", "");
                textarea.style.position = "fixed";
                textarea.style.opacity = "0";
                document.body.append(textarea);
                textarea.select();
                document.execCommand("copy");
                textarea.remove();
            }
            copyButton.textContent = "Copied";
            window.setTimeout(() => {
                copyButton.textContent = "Copy";
            }, 1600);
        });
    }

    // ---- Fade-in-up reveal ----
    const revealObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("visible");
                    revealObserver.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );
    document.querySelectorAll(".fade-in-up").forEach((el) => revealObserver.observe(el));

    // ---- Nav active state ----
    const sections = Array.from(document.querySelectorAll("main section[id]"));
    const navLinks = Array.from(document.querySelectorAll(".nav-links a"));
    const navObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                navLinks.forEach((link) => {
                    const href = ["feature-video", "long-video", "short-video"].includes(entry.target.id)
                        ? "#feature-video"
                        : `#${entry.target.id}`;
                    link.classList.toggle("active", link.getAttribute("href") === href);
                });
            });
        },
        { threshold: 0.34 }
    );
    sections.forEach((section) => navObserver.observe(section));

    // ---- Video preview modal ----
    const modal = document.querySelector(".modal");
    const modalVideo = document.querySelector(".modal-video");
    const modalTitle = document.querySelector(".modal-copy h3");
    const modalPrompt = document.querySelector(".modal-copy p");
    const closeModal = () => {
        if (!modal || !modalVideo) return;
        modal.classList.remove("is-open");
        modal.setAttribute("aria-hidden", "true");
        document.body.classList.remove("modal-open");
        modalVideo.pause();
        modalVideo.removeAttribute("src");
        modalVideo.load();
    };
    document.querySelectorAll(".case-media").forEach((media) => {
        media.addEventListener("click", () => {
            if (!modal || !modalVideo || !media.dataset.video) return;
            modalVideo.src = media.dataset.video;
            modalTitle.textContent = media.dataset.title || "JoyAI-Echo preview";
            modalPrompt.textContent = media.dataset.prompt || "";
            modal.classList.add("is-open");
            modal.setAttribute("aria-hidden", "false");
            document.body.classList.add("modal-open");
            modalVideo.play().catch(() => {});
        });
    });
    document.querySelectorAll("[data-close-modal]").forEach((el) => {
        el.addEventListener("click", closeModal);
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeModal();
    });
});
