# JSX → Obsidian 정적 HTML 변환 프로젝트

## 목적

**React(JSX) 기반 인터랙티브 HTML 문서**를 작성하고, 이를 **Obsidian에서 렌더링 가능한 순수 HTML+CSS 정적 파일**로 변환하여 관리합니다.

## 프로젝트 경로

`C:\Users\Lee chang yeoun\OneDrive\Desktop\Claude\HTML\`

---

## 워크플로우

```
JSX 작성/수정 (Claude AI)
        ↓
브라우저에서 확인 (.html 열기)
        ↓
정적 HTML로 변환 (Claude AI)
        ↓
Obsidian에서 열람 (HTML Reader 플러그인)
```

---

## 파일 목록

### 인터랙티브 원본 (브라우저용)
| 파일 | 설명 |
|---|---|
| `BQB_Beginner_Guide.html` | 초보자 가이드 — React 인터랙티브 |
| `BQB_Intermediate_Guide.html` | 중급 가이드 — React 인터랙티브 |

### Obsidian용 정적 변환
| 파일 | 설명 |
|---|---|
| `BQB_Beginner_Static.html` | 초보자 가이드 — 순수 HTML+CSS (Obsidian 호환) |

### Obsidian 마크다운
| 파일 | 설명 |
|---|---|
| `BQB_Beginner_Guide.md` | 초보자 가이드 — Obsidian 마크다운 (callout, mermaid 포함) |
| `BQB_Intermediate_Guide.md` | 중급 가이드 — Obsidian 마크다운 (callout, mermaid 포함) |

### 프로젝트 설정
| 파일 | 설명 |
|---|---|
| `CLAUDE.md` | Claude AI 지침서 (변환 규칙, 프롬프트 템플릿) |
| `README.md` | 이 파일 (프로젝트 설명서) |

---

## 환경 설정

### Obsidian 필수 설정
1. **설정 → 파일 및 링크 → "모든 파일 확장자 감지"** 켜기
2. **커뮤니티 플러그인 → HTML Reader** 설치 및 활성화

### 파일 형식별 용도
| 형식 | 용도 | 뷰어 |
|---|---|---|
| `.html` (JSX) | 인터랙티브 버전 (원본) | 웹 브라우저 |
| `_Static.html` | Obsidian 열람용 | Obsidian HTML Reader |
| `.md` | 빠른 참조/검색용 | Obsidian 기본 뷰어 |
| `.jsx` | 코드 편집용 원본 | 코드 에디터 |

---

## 변환 방법

### JSX → 정적 HTML 변환 요청 (Claude AI)

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

### 마크다운 변환 요청 (선택)

```
다음 HTML 파일의 콘텐츠를 Obsidian 마크다운으로 변환해줘.
Obsidian callout, mermaid 다이어그램, LaTeX 수식, 하이라이트(==텍스트==) 등
Obsidian 전용 기능을 최대한 활용해서 시각적으로 풍부하게 만들어줘.

[파일 경로 또는 코드 붙여넣기]
```
