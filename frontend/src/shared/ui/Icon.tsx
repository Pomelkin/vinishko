type Name =
  | "search"
  | "camera"
  | "menu"
  | "close"
  | "back"
  | "filter"
  | "upload"
  | "arrow"
  | "check"
  | "chevron";
const paths: Record<Name, React.ReactNode> = {
  search: (
    <>
      <circle cx="10.5" cy="10.5" r="6.5" />
      <path d="m16 16 4 4" />
    </>
  ),
  camera: (
    <>
      <path d="M8 6 9.5 3h5L16 6h4a1 1 0 0 1 1 1v12H3V7a1 1 0 0 1 1-1Z" />
      <circle cx="12" cy="12" r="3.5" />
    </>
  ),
  menu: <path d="M4 6h16M4 12h16M4 18h16" />,
  close: <path d="m6 6 12 12M6 18 18 6" />,
  back: <path d="m14 5-7 7 7 7M7 12h14" />,
  filter: <path d="M3 6h18M6 12h12M9 18h6" />,
  upload: (
    <>
      <path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5" />
    </>
  ),
  arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
  check: <path d="m4 12 5 5L20 6" />,
  chevron: <path d="m8 4 8 8-8 8" />,
};
export function Icon({ name, size = 22 }: { name: Name; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
