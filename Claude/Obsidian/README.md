# Obsidian HTML Viewer

Obsidian에서 React 기반 인터랙티브 HTML 가이드를 열어볼 수 있는 환경입니다.

---

## 구조

```
Claude/Obsidian/
├── html/                          # HTML 가이드 파일
├── JSX_Obsidian_변환프롬프트_v4.4.md  # JSX→HTML 변환 프롬프트
├── CLAUDE.md                      # Claude Code 지시사항
└── README.md
```

---

## 핵심 플러그인

| 플러그인 | 역할 | 필수 여부 |
|---------|------|----------|
| **obsidian-html-plugin** (HTML Reader) | `.html` 파일을 파일 탐색기에서 인식 | 필수 |
| **obsidian-custom-frames** | `file:///` 경로로 HTML을 WebView 렌더링 | 필수 |

> React CDN 기반 HTML은 HTML Reader만으로 렌더링되지 않습니다. **Custom Frames에 등록해야** 정상적으로 열립니다.

---

## 동작 원리

- Custom Frames의 `forceIframe: false` 설정으로 **Electron WebView**를 사용
- `file:///` 절대경로로 로컬 HTML 파일을 직접 로드
- WebView이므로 React/Babel CDN 스크립트 실행 가능
- **별도 서버 불필요** (localhost 서버 없이 동작)

---

## 시작하기

### 1. 커뮤니티 플러그인 설치

1. **설정** > **커뮤니티 플러그인** > **제한 모드 해제** > **찾아보기**
2. `HTML Reader`, `Custom Frames` 두 개 모두 설치 및 활성화

### 2. HTML 파일 열기

Custom Frames에 등록된 파일은 리본(좌측 사이드바) 아이콘을 클릭하거나, `Ctrl + P` 명령어 팔레트에서 검색하여 열 수 있습니다.

---

## Custom Frames에 새 HTML 등록하기

HTML 파일을 새로 만들면 **반드시 Custom Frames에 등록**해야 Obsidian에서 열 수 있습니다.

`.obsidian/plugins/obsidian-custom-frames/data.json`의 `frames` 배열에 추가:

```json
{
    "url": "file:///C:/Users/user/Desktop/Claud/Git/Undergraduate-research-student/Claude/Obsidian/html/파일명.html",
    "displayName": "표시 이름",
    "icon": "아이콘명",
    "hideOnMobile": true,
    "addRibbonIcon": true,
    "openInCenter": true,
    "zoomLevel": 1,
    "forceIframe": false,
    "customCss": "body { overflow-x: hidden; }",
    "customJs": ""
}
```

등록 후 **Obsidian 재시작** 필요.

> 한글 파일명은 URL 인코딩 필요: `센터` → `%EC%84%BC%ED%84%B0`

---

## HTML 가이드 만들기

JSX/TSX 파일을 Obsidian에서 실행 가능한 단일 HTML로 변환할 수 있습니다.
AI에게 [`JSX_Obsidian_변환프롬프트_v4.4.md`](./JSX_Obsidian_변환프롬프트_v4.4.md)를 붙여넣고 JSX 파일을 주면 자동으로 변환해줍니다.

---

## 참고 자료

- [obsidian-html-plugin (GitHub)](https://github.com/nuthrash/obsidian-html-plugin) — HTML Reader 플러그인
- [obsidian-custom-frames (GitHub)](https://github.com/Ellpeck/ObsidianCustomFrames) — Custom Frames 플러그인
