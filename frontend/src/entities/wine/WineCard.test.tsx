import { render, screen, fireEvent } from "@testing-library/react";
import { it, expect, vi } from "vitest";
import { WineCard, Ratings } from "./WineCard";
import { wines } from "../../mocks/catalog";
it("does not fabricate missing ratings and exposes an actionable card", () => {
  const click = vi.fn();
  render(
    <>
      <WineCard wine={wines[0]} onClick={click} />
      <Ratings wine={wines[0]} />
    </>,
  );
  expect(screen.queryByText(/Народный рейтинг/)).toBeNull();
  fireEvent.click(screen.getByRole("button"));
  expect(click).toHaveBeenCalledOnce();
});
