import { useEffect, useRef, type ReactNode } from "react";
import { Icon } from "./Icon";
import s from "./Sheet.module.css";
interface Props {
  children: ReactNode;
  title: string;
  onClose: () => void;
  footer?: ReactNode;
  initialScroll?: number;
  onScroll?: (y: number) => void;
  viewKey?: string;
  compact?: boolean;
}
export function Sheet({
  children,
  title,
  onClose,
  footer,
  initialScroll = 0,
  onScroll,
  viewKey,
  compact = false,
}: Props) {
  const dialog = useRef<HTMLDialogElement>(null),
    scroller = useRef<HTMLDivElement>(null),
    close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const node = dialog.current!,
      previous = document.activeElement as HTMLElement | null;
    const oldOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    node.showModal();
    const cancel = (e: Event) => {
      e.preventDefault();
      close.current();
    };
    node.addEventListener("cancel", cancel);
    return () => {
      node.removeEventListener("cancel", cancel);
      node.close();
      document.body.style.overflow = oldOverflow;
      if (previous?.isConnected) previous.focus({ preventScroll: true });
    };
  }, []);
  useEffect(() => {
    if (scroller.current) scroller.current.scrollTop = initialScroll;
  }, [viewKey]); // Restore only on a view change, never during scrolling.
  const gesture = useRef<{ x: number; y: number; eligible: boolean } | null>(
    null,
  );
  useEffect(() => {
    const node = dialog.current!;
    const start = (e: TouchEvent) => {
      const t = e.touches[0];
      gesture.current = {
        x: t.clientX,
        y: t.clientY,
        eligible:
          !!(e.target as Element).closest("[data-handle]") ||
          (scroller.current?.scrollTop ?? 0) <= 0,
      };
    };
    const move = (e: TouchEvent) => {
      const g = gesture.current;
      if (!g) return;
      const dy = e.touches[0].clientY - g.y,
        dx = e.touches[0].clientX - g.x;
      if (g.eligible && dy > 12 && Math.abs(dy) > Math.abs(dx))
        e.preventDefault();
    };
    const end = (e: TouchEvent) => {
      const g = gesture.current;
      gesture.current = null;
      if (!g) return;
      const t = e.changedTouches[0];
      if (g.eligible && t.clientY - g.y > 80 && Math.abs(t.clientX - g.x) < 100)
        close.current();
    };
    node.addEventListener("touchstart", start, { passive: true });
    node.addEventListener("touchmove", move, { passive: false });
    node.addEventListener("touchend", end);
    return () => {
      node.removeEventListener("touchstart", start);
      node.removeEventListener("touchmove", move);
      node.removeEventListener("touchend", end);
    };
  }, []);
  const pointerY = useRef<number | null>(null);
  return (
    <dialog
      ref={dialog}
      className={s.sheet}
      style={compact ? { maxHeight: "60dvh" } : undefined}
      aria-label={title}
      onClick={(e) => {
        if (e.target === dialog.current) {
          const r = dialog.current.getBoundingClientRect();
          if (e.clientY < r.top || e.clientX < r.left || e.clientX > r.right)
            onClose();
        }
      }}
    >
      <div
        data-handle
        className={s.handle}
        onPointerDown={(e) => {
          pointerY.current = e.clientY;
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerUp={(e) => {
          if (pointerY.current !== null && e.clientY - pointerY.current > 80)
            onClose();
          pointerY.current = null;
        }}
      >
        <span />
      </div>
      <button
        className={`icon-button ${s.close}`}
        aria-label="Закрыть окно"
        onClick={onClose}
      >
        <Icon name="close" />
      </button>
      <div
        ref={scroller}
        data-sheet-scroll
        className={s.content}
        onScroll={(e) => onScroll?.(e.currentTarget.scrollTop)}
      >
        {children}
      </div>
      {footer && <div className={s.footer}>{footer}</div>}
    </dialog>
  );
}
