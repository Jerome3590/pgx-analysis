## Goal

Turn one 4700‑line `index.html` into:

- A lean `index.html` that defines layout + tab navigation.
- Separate HTML files per tab, loaded on demand via JavaScript.

No build step, still pure static hosting on S3.[^1][^2]

## File layout

- `index.html` – main page (head, global CSS/JS, tab buttons, single content container).
- `tabs/overview.html` – HTML for Overview tab only.
- `tabs/analytics.html` – HTML for Analytics tab only.
- `tabs/settings.html` – HTML for Settings tab only.
- `tabs.js` – JS that loads tab files into the content container.[^3][^4]


## Index.html changes

1. Keep `<head>` as‑is (CSS links, global JS, meta).
2. Keep/define the **tab bar**:
```html
<div class="tabs">
  <button data-tab="overview">Overview</button>
  <button data-tab="analytics">Analytics</button>
  <button data-tab="settings">Settings</button>
</div>
```

3. Replace all inline tab sections with a **single container**:
```html
<div id="tab-content"></div>
<script src="tabs.js"></script>
```

So index.html becomes a small shell: nav + tab bar + one content area.[^5][^3]

## Per-tab files

```
Each tab file contains only that tab’s markup (no `<html>`, `<head>`, or `<body>`):
```

```html
<!-- tabs/overview.html -->
<section class="tab-panel">
  <h2>Overview</h2>
  <!-- overview content here -->
</section>
```

Same idea for `tabs/analytics.html`, `tabs/settings.html`, etc.[^3]

## Tab loading script

`tabs.js` handles click → fetch → inject:

```js
const TAB_MAP = {
  overview: "tabs/overview.html",
  analytics: "tabs/analytics.html",
  settings: "tabs/settings.html",
};

const tabCache = {};

async function loadTab(name) {
  const file = TAB_MAP[name];
  if (!file) return;

  // use cache if available
  if (tabCache[name]) {
    document.getElementById("tab-content").innerHTML = tabCache[name];
    return;
  }

  try {
    const res = await fetch(file);
    if (!res.ok) throw new Error(res.statusText);
    const html = await res.text();
    tabCache[name] = html;
    document.getElementById("tab-content").innerHTML = html;
  } catch (e) {
    console.error("Failed to load tab", name, e);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tabs [data-tab]").forEach(btn => {
    btn.addEventListener("click", () => loadTab(btn.dataset.tab));
  });

  // load default tab
  loadTab("overview");
});
```

This keeps extra requests minimal (one per tab, cached in memory) and works with S3 static hosting.[^6][^7]

## What you gain

- `index.html` shrinks dramatically and is easier to reason about.
- Each tab’s code is isolated in its own file.
- Heavy/rarely used tabs can be lazy-loaded; main view can render faster.[^8][^9]

Would you like a concrete before/after snippet using your exact tab markup style (e.g., Bootstrap tabs or your current custom tabs)?

<div align="center">⁂</div>

[^1]: https://stackoverflow.com/questions/53462902/aws-whats-the-recommended-way-to-host-the-client-side-html-css-js-image-f

[^2]: https://www.reddit.com/r/aws/comments/jdj2ox/can_i_host_a_javascript_website_using_s3/

[^3]: https://stackoverflow.com/questions/49918732/html-tabs-with-content-in-separate-files

[^4]: https://dev.to/staa99/partial-views-templates-in-plain-html-27ni

[^5]: https://www.w3schools.com/howto/howto_js_tabs.asp

[^6]: https://stackoverflow.com/questions/57987543/how-do-i-use-the-fetch-api-to-load-html-page-with-its-javascript

[^7]: https://stackoverflow.com/questions/51808949/improving-performance-of-multiple-fetch-requests-in-javascript

[^8]: https://documentation.platformos.com/best-practices/frontend-performance/fragment-lazy-loading

[^9]: https://web.dev/learn/performance/optimize-resource-loading

