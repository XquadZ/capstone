/* Hoseo LENS — 신규 공지 Windows 알림 클릭 → 공지 페이지 열기 */
self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const data = event.notification.data || {};
  const url = String(data.url || "").trim();

  event.waitUntil(
    (async () => {
      const all = await self.clients.matchAll({
        type: "window",
        includeUncontrolled: true,
      });

      if (url.startsWith("http://") || url.startsWith("https://")) {
        for (const client of all) {
          if (client.url === url && "focus" in client) {
            return client.focus();
          }
        }
        if (self.clients.openWindow) {
          return self.clients.openWindow(url);
        }
      }

      // URL 없으면 LENS 탭이라도 포커스
      for (const client of all) {
        if ("focus" in client) return client.focus();
      }
      return undefined;
    })()
  );
});
