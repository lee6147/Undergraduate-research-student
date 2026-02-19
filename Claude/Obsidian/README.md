<div align="center">

# ⚛️ AI JSX → Obsidian 완전 가이드

**AI로 만든 React 앱을 Obsidian에서 바로 실행하는 방법**

[![Obsidian](https://img.shields.io/badge/Obsidian-7C3AED?style=for-the-badge&logo=obsidian&logoColor=white)](https://obsidian.md)
[![React](https://img.shields.io/badge/React_18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

```
🤖 AI에게 요청  →  📄 .jsx 파일  →  🔄 HTML 변환  →  🔌 플러그인 등록  →  🚀 Obsidian 실행!
```

</div>

---

## 📖 목차

- [왜 이 과정이 필요할까?](#-왜-이-과정이-필요할까)
- [STEP 1 — Custom Frames 플러그인 설치](#-step-1--custom-frames-플러그인-설치)
- [STEP 2 — AI로 JSX 파일 만들기](#-step-2--ai로-jsx-파일-만들기)
- [STEP 3 — JSX를 HTML로 변환하기](#-step-3--jsx를-html로-변환하기)
- [STEP 4 — Obsidian에 HTML 등록하기](#-step-4--obsidian에-html-등록하기)
- [STEP 5 — 실행!](#-step-5--실행)
- [FAQ](#-faq)
- [문제 해결](#-문제-해결)
- [프로젝트 구조](#-프로젝트-구조)

---

## 🤔 왜 이 과정이 필요할까?

Obsidian은 기본적으로 **마크다운 에디터**입니다. HTML이나 JavaScript를 직접 실행하지 못합니다.

| 방법 | JS 실행 | React 앱 | 설명 |
|:----:|:-------:|:--------:|------|
| Obsidian 기본 | ❌ | ❌ | HTML을 첨부파일로만 인식 |
| HTML Reader 플러그인 | ❌ | ❌ | HTML은 보이지만 JS가 차단됨 |
| **✨ Custom Frames** | **✅** | **✅** | **Electron WebView로 모든 것이 동작!** |

> 💡 **Custom Frames**는 Obsidian 안에 작은 브라우저(WebView)를 열어줍니다.
> 일반 브라우저에서 되는 건 전부 됩니다 — React, Vue, 게임 등 뭐든!

---

## 🔌 STEP 1 — Custom Frames 플러그인 설치

### 1-1. 커뮤니티 플러그인 활성화

> ⚠️ 이미 활성화되어 있다면 건너뛰세요!

1. Obsidian 열기 → 좌측 하단 **⚙️ 톱니바퀴** (설정) 클릭
2. 왼쪽 메뉴에서 **"Community plugins"** 클릭
3. **"Turn on community plugins"** 클릭 → 경고창에서 **"Turn on"**

### 1-2. Custom Frames 설치

1. 설정 → Community plugins → **"Browse"** 클릭
2. 검색창에 **`Custom Frames`** 입력
3. **"Custom Frames"** by lishid 선택
4. **"Install"** 클릭 → **"Enable"** 클릭

> ✅ 설정 → Community plugins 목록에 **Custom Frames**가 보이고 토글이 켜져 있으면 성공!

<details>
<summary>📦 <b>선택사항: obsidian-git 플러그인</b></summary>

<br/>

Vault를 GitHub에 자동 백업하고 싶다면 같은 방법으로 설치합니다.

1. 설정 → Community plugins → Browse
2. **"Git"** 검색 → **"Git"** by Vinzent03 선택
3. Install → Enable

설치 후 Vault에서 `git init` → GitHub 원격 저장소 연결 → 자동 백업 간격 설정

</details>

---

## 🤖 STEP 2 — AI로 JSX 파일 만들기

### AI에게 이렇게 요청하세요

```
React로 "학습 퀴즈 앱"을 만들어줘.

조건:
- 하나의 파일에 모든 컴포넌트 작성
- useState, useEffect 등 React Hooks 사용
- 외부 라이브러리 없이 React만 사용
- export default function App() 형태로 작성
- 다크 테마 UI
```

> 🎯 **핵심:** `"하나의 파일"`과 `"외부 라이브러리 없이"`를 꼭 명시하세요.
> 여러 파일로 나뉘거나 npm 패키지를 쓰면 변환이 복잡해집니다.

### 파일로 저장

AI가 만들어준 코드를 **`.jsx`** 파일로 저장합니다.

```
파일명:   MyQuizApp.jsx
저장 위치: 어디든 OK (바탕화면, 프로젝트 폴더 등)
```

### 정상적인 JSX 파일 구조

```jsx
import { useState, useEffect } from "react";   // ← 이 줄이 있어야 함

function QuizQuestion() {
  return <div>...</div>;
}

export default function App() {                 // ← 이것도 있어야 함
  const [score, setScore] = useState(0);
  return (
    <div>
      <QuizQuestion />
    </div>
  );
}
```

---

## 🔄 STEP 3 — JSX를 HTML로 변환하기

JSX는 브라우저가 직접 읽을 수 없으므로 **단독 실행 HTML**로 변환해야 합니다.

### ⚡ 방법 A: Claude Code 자동 변환 (추천!)

```
"이 JSX 파일을 Obsidian에서 실행 가능하게 해줘"
```

> 🎉 변환 + 등록 + Obsidian 재시작까지 **자동으로** 해줍니다!

### ✋ 방법 B: 수동 변환

3가지만 바꾸면 됩니다:

#### ① import 문 교체

```diff
- import { useState, useEffect, useRef } from "react";
+ const { useState, useEffect, useRef } = React;
```

#### ② export default 제거

```diff
- export default function App() {
+ function App() {
```

#### ③ HTML 템플릿으로 감싸기

```html
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>앱 제목</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { overflow-x: hidden; background: #030712; font-family: system-ui, sans-serif; }
</style>
<script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
</head>
<body>
<div id="root"></div>
<script type="text/babel">
const { useState, useEffect, useRef } = React;

/* 여기에 변환된 JSX 코드를 붙여넣기 */

ReactDOM.createRoot(document.getElementById("root")).render(
  React.createElement(App)
);
</script>
</body>
</html>
```

> 🧪 **변환 확인:** HTML 파일을 브라우저(Chrome, Edge)에서 더블클릭해서 열어보세요.
> 브라우저에서 안 되면 Obsidian에서도 안 됩니다!

---

## ⚙️ STEP 4 — Obsidian에 HTML 등록하기

### Custom Frames 설정

1. Obsidian 설정(⚙️) → 왼쪽 메뉴 스크롤 → **"Custom Frames"** 클릭
2. **"Add Frame"** 또는 **"+"** 버튼 클릭
3. 아래 항목을 채워 넣기:

| 설정 항목 | 입력값 | 설명 |
|:---------:|--------|------|
| **URL** | `file:///C:/경로/파일.html` | HTML 파일의 절대경로 |
| **Display Name** | 앱 이름 | Obsidian에서 표시될 이름 |
| **Open in Center** | ✅ 체크 | 중앙 패널에서 열기 |
| **⚡ Force Iframe** | **❌ 반드시 해제!** | 이거 안 끄면 안 됨! |

### 🚨 Force Iframe — 반드시 꺼야 합니다!

| | Force Iframe ON | Force Iframe OFF |
|:--:|:---:|:---:|
| 렌더링 방식 | iframe (제한적) | **WebView (완전)** |
| JavaScript | ❌ 차단됨 | ✅ 실행됨 |
| React 앱 | ❌ 안 됨 | ✅ **됨!** |

> 🚨 **가장 흔한 실수!** Force Iframe이 켜져 있으면 화면이 하얗게만 뜹니다.

### URL 작성 규칙

```bash
# ✅ 올바른 예시
file:///C:/Users/user/Desktop/MyApp.html
file:///D:/Projects/app.html
file:///C:/My%20Folder/app.html          # 공백 = %20

# ❌ 틀린 예시
file:///C:\Users\user\MyApp.html         # 역슬래시 사용
C:/Users/user/MyApp.html                 # file:/// 누락
```

> 📂 HTML 파일은 **Vault 안에 없어도 됩니다!** 컴퓨터 어디에든 있으면 경로만 맞추면 OK.

<details>
<summary>🔧 <b>고급: data.json 직접 편집</b></summary>

<br/>

설정 UI 대신 파일을 직접 편집할 수도 있습니다.

**파일 위치:** `{Vault}/.obsidian/plugins/obsidian-custom-frames/data.json`

```json
{
    "url": "file:///C:/Users/user/Desktop/MyApp.html",
    "displayName": "My App",
    "icon": "rocket",
    "hideOnMobile": true,
    "addRibbonIcon": true,
    "openInCenter": true,
    "zoomLevel": 1,
    "forceIframe": false,
    "customCss": "body { overflow-x: hidden; }",
    "customJs": ""
}
```

</details>

---

## 🚀 STEP 5 — 실행!

### 1. Obsidian 재시작

설정 변경 후 **반드시 Obsidian을 완전히 종료 후 다시 실행**해야 합니다.

> 설정이 메모리에 캐시되어 있어서 단순 새로고침으로는 반영되지 않습니다.

### 2. 앱 열기

```
Ctrl + P  →  등록한 이름 입력  →  클릭!
```

### ✅ 완료 체크리스트

- [ ] Custom Frames 플러그인 설치됨
- [ ] JSX → HTML 변환 완료
- [ ] 브라우저에서 HTML 테스트 성공
- [ ] Custom Frames에 URL 등록
- [ ] Force Iframe OFF 확인
- [ ] Obsidian 재시작
- [ ] Ctrl+P → 앱 실행 성공!

---

## ❓ FAQ

<details>
<summary><b>HTML 파일이 꼭 Vault 폴더 안에 있어야 하나요?</b></summary>

<br/>

**아니요!** `file:///` URL로 절대경로를 지정하면 컴퓨터 어디에 있든 열 수 있습니다.
바탕화면, D드라이브, 외장하드 등 어디든 OK.

</details>

<details>
<summary><b>JSX 없이 순수 HTML만 있으면요?</b></summary>

<br/>

그대로 Custom Frames에 등록하면 됩니다. 변환 과정이 필요 없습니다.

</details>

<details>
<summary><b>React 말고 Vue, Svelte도 되나요?</b></summary>

<br/>

**네!** CDN에서 해당 프레임워크를 로드하는 단독 HTML로 만들면 동일하게 작동합니다.
Custom Frames의 WebView는 일반 브라우저와 동일하게 동작합니다.

</details>

<details>
<summary><b>인터넷이 없으면 작동하나요?</b></summary>

<br/>

최초 1회는 CDN에서 React/Babel을 다운로드해야 합니다.
이후 브라우저 캐시에 저장되므로 대부분 오프라인에서도 작동합니다.

완전한 오프라인 지원이 필요하면 React/Babel `.js` 파일을 로컬에 저장하고 `script src` 경로를 수정하세요.

</details>

<details>
<summary><b>모바일 Obsidian에서도 되나요?</b></summary>

<br/>

Custom Frames는 **데스크탑 전용**입니다. 모바일에서는 작동하지 않습니다.

</details>

---

## 🔧 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| 화면이 하얗게 뜸 | Force Iframe이 켜져 있음 | 설정에서 **Force Iframe OFF** |
| 설정 반영 안 됨 | Obsidian 캐시 | **완전 종료** 후 재시작 (최소화 X) |
| "Failed to load" | 경로 오류 | `file:///` + 정방향 슬래시(`/`) 확인 |
| React 에러 | 변환 누락 | `import` → `const` 변환, `export default` 제거 확인 |
| 콘솔 에러 | JS 문법 오류 | 브라우저에서 F12 → Console 탭 확인 |

---

## 📁 프로젝트 구조

```
📦 Claud/
├── 📄 README.md            ← 이 파일 (GitHub 가이드)
├── 📄 README.html          ← 인터랙티브 가이드 (애니메이션 포함)
├── 📄 CLAUDE.md            ← Claude Code 자동화 지침
└── 📂 Quantum/             ← 학습 앱 모음
    ├── 📄 BQB_Final_Launcher.jsx    ← 원본 JSX
    └── 📄 BQB_Final_Launcher.html   ← 변환된 HTML
```

---

<div align="center">

Made with ⚛️ for Obsidian lovers

**Custom Frames** by lishid · **obsidian-git** by Vinzent03

</div>
