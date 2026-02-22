# Obsidian HTML Viewer

Obsidian에서 HTML 가이드 파일을 바로 열어볼 수 있는 환경입니다.
별도 설정 없이 레포를 클론하고 Obsidian에서 열면 바로 사용할 수 있습니다.

---

## 핵심 플러그인

| 플러그인 | 역할 | 필수 여부 |
|---------|------|----------|
| **obsidian-html-plugin** (HTML Reader) | `.html` 파일을 파일 탐색기에서 인식 → 클릭으로 바로 열기 | 필수 |
| **obsidian-custom-frames** | HTML 파일을 좌측 리본 아이콘 바로가기로 등록 | 선택 |

> Obsidian은 기본적으로 `.html` 파일을 열 수 없습니다. `obsidian-html-plugin`이 이를 가능하게 합니다.

---

## 시작하기

### 1. 레포 클론

```bash
git clone https://github.com/lee6147/Undergraduate-research-student.git
```

### 2. Obsidian에서 볼트 열기

1. Obsidian 실행
2. **다른 보관함 열기** → **폴더를 보관함으로 열기**
3. 클론한 폴더 내 `Claude/Obsidian` 경로를 선택

### 3. 커뮤니티 플러그인 활성화

1. **설정**(좌측 하단 톱니바퀴) → **커뮤니티 플러그인**
2. **제한 모드 해제** 클릭
3. 설치된 플러그인 목록에서 **HTML Reader** 활성화 확인

### 4. HTML 파일 열기

파일 탐색기에서 `Guide_HTML/` 폴더 안의 `.html` 파일을 클릭하면 바로 열립니다.

---

## 볼트 구조

```
Claude/Obsidian/                   ← 볼트 루트
├── .obsidian/
│   └── plugins/
│       ├── obsidian-html-plugin/  ← HTML 뷰어 (필수)
│       ├── obsidian-custom-frames/← 리본 바로가기 (선택)
│       └── obsidian-git/
├── Guide_HTML/                    ← HTML 가이드 보관 폴더
│   ├── obsidian_guide.html
│   ├── Git_guide.html
│   ├── BQB_Final_Launcher.html
│   └── RnD_센터_구축_가이드.html
└── *.md
```

---

## HTML 파일 추가하기

### 방법 1: 폴더에 넣기 (간단)

`Guide_HTML/` 폴더에 `.html` 파일을 넣으면 끝입니다.
`obsidian-html-plugin`이 자동 인식하므로 별도 등록이 필요 없습니다.

### 방법 2: 리본 바로가기 등록 (Custom Frames)

`.obsidian/plugins/obsidian-custom-frames/data.json`의 `frames` 배열에 추가:

```json
{
    "url": "file:///C:/Users/user/Desktop/Claud/Obsidian/Guide_HTML/파일명.html",
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

| 필드 | 설명 |
|------|------|
| `url` | `file:///` + HTML 절대 경로 (한글은 URL 인코딩) |
| `displayName` | Obsidian에 표시될 이름 |
| `icon` | Lucide 아이콘 이름 (`rocket`, `book`, `git-branch`, `building` 등) |
| `addRibbonIcon` | `true`면 좌측 리본에 바로가기 아이콘 생성 |

> 한글 파일명 URL 인코딩 예시: `센터` → `%EC%84%BC%ED%84%B0`

---

## 등록된 HTML 가이드

| 파일 | 표시명 | 아이콘 |
|------|-------|--------|
| `obsidian_guide.html` | Obsidian Guide | book |
| `Git_guide.html` | Git 완전 정복 가이드 | git-branch |
| `BQB_Final_Launcher.html` | BQB Final Launcher | rocket |
| `RnD_센터_구축_가이드.html` | R&D 센터 구축 가이드 | building |

---

## 경로 변경 시 주의사항

| 상황 | HTML Plugin (파일 탐색기) | Custom Frames (리본 아이콘) |
|------|--------------------------|---------------------------|
| 파일명 변경 | 자동 반영 | `data.json` URL 수정 필요 |
| 다른 폴더로 이동 | 자동 반영 (볼트 내) | `data.json` URL 수정 필요 |
| 볼트 밖으로 이동 | 인식 불가 | `data.json` URL 수정 필요 |
| 새 HTML 추가 | 자동 인식 | `data.json`에 등록 필요 |

---

## HTML 가이드 만들기

JSX/TSX 파일을 Obsidian에서 실행 가능한 단일 HTML로 변환할 수 있습니다.
AI에게 [`JSX_Obsidian_변환프롬프트_v4.4.md`](./JSX_Obsidian_변환프롬프트_v4.4.md)를 붙여넣고 JSX 파일을 주면 자동으로 변환해줍니다.

---

## 참고 자료

- [obsidian-html-plugin (GitHub)](https://github.com/nuthrash/obsidian-html-plugin) — HTML Reader 플러그인 원본
