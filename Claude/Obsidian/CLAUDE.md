# CLAUDE.md — 프로젝트 지침서

## 프로젝트 개요

이 폴더는 **React(JSX) 기반 인터랙티브 HTML 문서**를 작성하고, 이를 **Obsidian에서 렌더링 가능한 순수 HTML+CSS 정적 파일**로 변환하는 프로젝트입니다.

---

## 워크플로우

### 1단계: JSX 코드 작성/수정
- React + Babel CDN 기반의 `.jsx` 또는 `.html` 파일로 인터랙티브 콘텐츠를 작성합니다.
- 이 파일은 **브라우저에서 실행**되며, 버튼/탭/애니메이션/시뮬레이션 등 인터랙티브 요소를 포함합니다.

### 2단계: Obsidian 호환 HTML로 변환
- JSX 파일의 콘텐츠를 **순수 HTML+CSS**로 변환합니다.
- 변환된 파일은 Obsidian의 **HTML Reader 플러그인**에서 렌더링됩니다.

---

## 변환 규칙 (JSX → 정적 HTML)

### 필수 제거 항목
- `<script>` 태그 전부 (React, ReactDOM, Babel, 모든 JavaScript)
- 외부 CDN 참조 (`unpkg.com`, `cdn.jsdelivr.net` 등)
- `useState`, `useEffect`, `useRef` 등 React hooks
- `onClick`, `onChange` 등 이벤트 핸들러
- 동적 렌더링 로직 (조건부 렌더링, `.map()` 등)

### 필수 유지 항목
- 모든 텍스트 콘텐츠 (한 글자도 빠짐없이)
- 시각적 디자인 (색상, 레이아웃, 카드 구조, 다크 테마)
- SVG 다이어그램 (정적 버전으로 변환)
- CSS 스타일 (`<style>` 태그 내 인라인 또는 내장 CSS)
- 반응형 레이아웃 (`@media` 쿼리)

### 변환 전략
| JSX 요소 | 정적 HTML 변환 방식 |
|---|---|
| 탭 전환 (`useState`로 탭 관리) | 모든 탭 콘텐츠를 **한 페이지에 순서대로** 펼침 |
| 접기/펼치기 버튼 | 펼쳐진 상태로 고정 |
| 애니메이션/시뮬레이션 | 대표 상태의 **정적 스냅샷** (SVG 또는 ASCII) |
| 조건부 렌더링 (`{show && ...}`) | 모든 상태를 **동시에 표시** |
| React 컴포넌트 | 해당 JSX를 **순수 HTML `<div>`로 풀어서** 작성 |
| 인라인 스타일 (`style={{...}}`) | HTML `style="..."` 속성으로 변환 |
| CSS-in-JS 변수 (`const C = {...}`) | `<style>` 태그 내 CSS 변수 또는 직접 색상값 사용 |

### 파일 명명 규칙
- JSX 원본: `파일명.html` 또는 `파일명.jsx`
- Obsidian용 정적 변환: `파일명_Static.html`

---

## 파일 구조

```
HTML/
├── .obsidian/              # Obsidian 볼트 설정
├── 파일명.html             # JSX 원본 (브라우저용 인터랙티브)
├── 파일명.jsx              # JSX 원본 (코드 편집용)
├── 파일명_Static.html      # Obsidian용 정적 HTML
├── 파일명.md               # Obsidian용 마크다운 (선택)
├── CLAUDE.md               # 이 파일 (Claude AI 지침서)
└── README.md               # 프로젝트 설명서
```

---

## 변환 시 주의사항

1. **콘텐츠 누락 금지**: JSX 원본의 모든 텍스트, 수치, 표, 비유가 정적 HTML에 반드시 포함되어야 합니다.
2. **디자인 충실도**: 색상 코드, 카드 레이아웃, 다크 테마를 원본과 최대한 동일하게 유지합니다.
3. **자기완결 파일**: 외부 리소스(CDN, 웹폰트, 이미지 URL) 없이 단일 `.html` 파일로 완결되어야 합니다.
4. **Obsidian HTML Reader 호환**: JavaScript가 전혀 없어야 합니다. CSS 애니메이션(`@keyframes`)은 허용될 수 있으나, 필수가 아니면 제외합니다.
5. **SVG 다이어그램**: 복잡한 React 컴포넌트로 생성되던 그래프/다이어그램은 인라인 `<svg>`로 정적 변환합니다.
6. **인코딩**: 한글 콘텐츠이므로 `<meta charset="UTF-8" />`을 `<head>` 안에 반드시 포함합니다.
7. **저장 경로**: 변환된 파일은 원본과 같은 폴더(`C:\Users\Lee chang yeoun\OneDrive\Desktop\Claude\HTML\`)에 저장합니다.

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| Obsidian 사이드바에 HTML 파일이 안 보임 | "모든 파일 확장자 감지" 꺼짐 | 설정 → 파일 및 링크 → 켜기 |
| HTML 파일이 코드로 보임 (렌더링 안 됨) | HTML Reader 플러그인 미설치/비활성 | 커뮤니티 플러그인에서 설치 후 활성화 |
| 렌더링은 되지만 빈 화면 | JavaScript 의존 코드가 남아있음 | `<script>` 태그가 완전히 제거됐는지 확인 |
| 한글이 깨짐 | charset 선언 누락 | `<meta charset="UTF-8" />` 추가 |
| 외부 이미지/폰트 안 보임 | CDN 참조 남아있음 | 외부 URL 제거, 인라인 또는 시스템 폰트 사용 |

---

## 프롬프트 템플릿

JSX → 정적 HTML 변환을 요청할 때 아래 프롬프트를 사용하세요:

```
다음 JSX 파일을 Obsidian HTML Reader에서 렌더링 가능한 순수 HTML+CSS 파일로 변환해줘.

변환 규칙:
- JavaScript/React/Babel 완전 제거 (script 태그 전부 삭제)
- 외부 CDN 참조 없이 단일 파일로 자기완결
- 모든 텍스트 콘텐츠 빠짐없이 포함
- 탭/버튼으로 숨겨진 콘텐츠는 한 페이지에 모두 펼쳐서 표시
- 원본의 다크 테마, 색상, 카드 레이아웃 유지
- SVG 다이어그램은 정적 인라인 SVG로 변환
- 파일명은 원본명_Static.html로 저장

[JSX 파일 경로 또는 코드 붙여넣기]
```
