const CACHE_NAME = 'potato-dashboard-v2'
// ponytail: only pre-cache the entrypoint. Hashed assets are cached at runtime
// on first load, so the shell works offline without hard-coding build hashes.
// /dashboard (not /) is the SPA shell — / serves the JSON discovery doc to
// non-browser Accept headers, which cache.addAll would otherwise store.
const SHELL_ASSETS = ['/dashboard', '/index.html', '/manifest.json', '/icon-192.png', '/icon-512.png']

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)).then(() => self.skipWaiting())
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => keys.filter((k) => k !== CACHE_NAME))
      .then((toDelete) => Promise.all(toDelete.map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  )
})

self.addEventListener('fetch', (event) => {
  // ponytail: only cache the app shell; never cache API calls, SSE, or uploads.
  const url = new URL(event.request.url)
  const isApi =
    url.pathname.startsWith('/v1/') ||
    url.pathname.startsWith('/admin/') ||
    url.pathname.startsWith('/analytics/') ||
    url.pathname.startsWith('/auth/') ||
    url.pathname.startsWith('/accounts/') ||
    url.pathname.startsWith('/health') ||
    url.pathname.startsWith('/stats') ||
    url.pathname.startsWith('/ladder') ||
    url.pathname.startsWith('/catalog') ||
    url.pathname.startsWith('/chat/api') ||
    url.pathname.startsWith('/preferences') ||
    url.protocol === 'chrome-extension:' ||
    url.pathname === '/analytics/events'
  if (event.request.method !== 'GET' || isApi) return

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.ok) {
          const clone = response.clone()
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone))
        }
        return response
      })
      .catch(() =>
        caches.match(event.request).then((cached) => {
          // Offline fallback to cached shell when the browser is disconnected.
          if (cached) return cached
          return caches.match('/index.html')
        })
      )
  )
})
