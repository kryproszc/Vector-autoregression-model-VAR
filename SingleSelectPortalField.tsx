import { ChevronDown } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

type SingleSelectPortalFieldProps = {
	label: string;
	value: string;
	options: string[] | Array<{ value: string; label: string }>;
	placeholder: string;
	onChange: (next: string) => void;
	disabled?: boolean;
	invalid?: boolean;
	errorMessage?: string | null;
	enableSearch?: boolean;
	searchPlaceholder?: string;
	noResultsText?: string;
};

export function SingleSelectPortalField({
	label,
	value,
	options,
	placeholder,
	onChange,
	disabled = false,
	invalid = false,
	errorMessage = null,
	enableSearch = false,
	searchPlaceholder = "Wyszukaj...",
	noResultsText = "Brak wyników.",
}: SingleSelectPortalFieldProps) {
	const baseOptions = options.map((option) =>
		typeof option === "string"
			? { value: option, label: option }
			: { value: option.value, label: option.label },
	);
	const normalizedOptions =
		value && !baseOptions.some((option) => option.value === value)
			? [{ value, label: value }, ...baseOptions]
			: baseOptions;

	const selectedLabel =
		normalizedOptions.find((option) => option.value === value)?.label ?? value;
	const MAX_VISIBLE_OPTIONS = 5;
	const OPTION_ROW_HEIGHT_ESTIMATE = 42;
	const POPUP_VERTICAL_PADDING = 12;
	const POPUP_MIN_HEIGHT = 120;
	const POPUP_GAP = 8;
	const visibleOptionsCount = Math.min(
		MAX_VISIBLE_OPTIONS,
		Math.max(1, normalizedOptions.length + (value ? 1 : 0)),
	);
	const estimatedPopupHeight =
		visibleOptionsCount * OPTION_ROW_HEIGHT_ESTIMATE + POPUP_VERTICAL_PADDING;

	const [isOpen, setIsOpen] = useState(false);
	const [searchQuery, setSearchQuery] = useState("");
	const triggerRef = useRef<HTMLButtonElement | null>(null);
	const popupRef = useRef<HTMLDivElement | null>(null);
	const [popupPosition, setPopupPosition] = useState<{
		top: number;
		left: number;
		width: number;
		maxHeight: number;
	} | null>(null);

	const updatePopupPosition = () => {
		const trigger = triggerRef.current;
		if (!trigger) {
			return;
		}

		const rect = trigger.getBoundingClientRect();
		const viewportPadding = 8;
		const dialog = trigger.closest('[role="dialog"]') as HTMLElement | null;
		const dialogRect = dialog?.getBoundingClientRect() ?? null;
		const availableTop = Math.max(
			viewportPadding,
			dialogRect ? dialogRect.top + viewportPadding : viewportPadding,
		);
		const availableBottom = Math.min(
			window.innerHeight - viewportPadding,
			dialogRect
				? dialogRect.bottom - viewportPadding
				: window.innerHeight - viewportPadding,
		);
		const maxViewportHeight = Math.max(
			POPUP_MIN_HEIGHT,
			window.innerHeight - viewportPadding * 2,
		);
		const desiredHeight = Math.max(
			POPUP_MIN_HEIGHT,
			Math.min(estimatedPopupHeight, maxViewportHeight),
		);
		const spaceBelow = Math.max(0, availableBottom - rect.bottom - POPUP_GAP);
		const spaceAbove = Math.max(0, rect.top - availableTop - POPUP_GAP);
		const shouldOpenUp =
			spaceBelow < Math.min(desiredHeight, 180) && spaceAbove > spaceBelow;
		const availableHeightOnChosenSide = shouldOpenUp ? spaceAbove : spaceBelow;
		const minPopupHeight = 96;
		const requestedHeight = Math.max(
			minPopupHeight,
			Math.min(
				desiredHeight,
				maxViewportHeight,
				Math.max(availableHeightOnChosenSide, minPopupHeight),
			),
		);
		const requestedTop = shouldOpenUp
			? rect.top - requestedHeight - POPUP_GAP
			: rect.bottom + POPUP_GAP;
		const minTop = availableTop;
		const maxTop = Math.max(minTop, availableBottom - requestedHeight);

		setPopupPosition({
			top: Math.min(Math.max(requestedTop, minTop), maxTop),
			left: Math.min(
				Math.max(viewportPadding, rect.left),
				window.innerWidth - rect.width - viewportPadding,
			),
			width: rect.width,
			maxHeight: requestedHeight,
		});
	};

	const normalizedQuery = searchQuery.trim().toLowerCase();
	const visibleOptions =
		enableSearch && normalizedQuery
			? normalizedOptions.filter((option) =>
					option.label.toLowerCase().includes(normalizedQuery),
				)
			: normalizedOptions;
	const optionListMaxHeight = Math.max(
		96,
		(popupPosition?.maxHeight ?? estimatedPopupHeight) -
			(enableSearch ? 54 : 0) -
			(value ? 44 : 0) -
			8,
	);

	useEffect(() => {
		if (!isOpen) {
			setPopupPosition(null);
			setSearchQuery("");
			return;
		}

		updatePopupPosition();
		const frameId = window.requestAnimationFrame(() => {
			updatePopupPosition();
		});
		const handleAnyScroll = () => {
			updatePopupPosition();
		};
		window.addEventListener("resize", updatePopupPosition);
		window.addEventListener("scroll", handleAnyScroll, true);

		let resizeObserver: ResizeObserver | null = null;
		if (typeof ResizeObserver !== "undefined") {
			resizeObserver = new ResizeObserver(() => {
				updatePopupPosition();
			});

			if (triggerRef.current) {
				resizeObserver.observe(triggerRef.current);
			}
			const dialog = triggerRef.current?.closest('[role="dialog"]');
			if (dialog instanceof HTMLElement) {
				resizeObserver.observe(dialog);
			}
		}

		return () => {
			window.cancelAnimationFrame(frameId);
			window.removeEventListener("resize", updatePopupPosition);
			window.removeEventListener("scroll", handleAnyScroll, true);
			resizeObserver?.disconnect();
		};
	}, [isOpen, normalizedOptions.length, visibleOptions.length, value, searchQuery]);

	useEffect(() => {
		if (!isOpen) {
			return;
		}

		const handlePointerDown = (event: MouseEvent) => {
			const target = event.target as Node | null;
			if (!target) {
				return;
			}

			const isInsideTrigger =
				triggerRef.current && triggerRef.current.contains(target);
			const isInsidePopup = popupRef.current && popupRef.current.contains(target);

			if (!isInsideTrigger && !isInsidePopup) {
				setIsOpen(false);
			}
		};

		document.addEventListener("mousedown", handlePointerDown);
		return () => {
			document.removeEventListener("mousedown", handlePointerDown);
		};
	}, [isOpen]);

	return (
		<label className="block text-slate-700 text-sm">
			<span className="mb-1 block">{label}</span>
			<button
				ref={triggerRef}
				type="button"
				disabled={disabled}
				onClick={() => {
					if (disabled) {
						return;
					}
					setIsOpen((prev) => !prev);
				}}
				className={`flex w-full items-start justify-between gap-2 rounded-lg border bg-white px-3 py-2 text-left text-sm outline-none transition-colors disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-700 ${
					invalid ? "border-rose-300 focus:border-rose-400" : "border-slate-300 focus:border-blue-400"
				}`}
			>
				<span className="min-w-0 whitespace-normal break-words">
					{selectedLabel || placeholder}
				</span>
				<ChevronDown size={14} className="text-slate-500" />
			</button>

			{isOpen && popupPosition
				? createPortal(
						<div
							ref={popupRef}
							className="fixed z-[80] overflow-hidden rounded-xl border border-slate-200 bg-white py-1.5 shadow-[0_14px_34px_rgba(15,23,42,0.14)]"
							style={{
								top: popupPosition.top,
								left: popupPosition.left,
								width: popupPosition.width,
								maxHeight: popupPosition.maxHeight,
							}}
						>
							{enableSearch ? (
								<div className="px-2 pb-2">
									<input
										type="text"
										value={searchQuery}
										onChange={(event) => setSearchQuery(event.target.value)}
										placeholder={searchPlaceholder}
										className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 text-slate-900 text-sm outline-none focus:border-blue-400"
									/>
								</div>
							) : null}
							{value ? (
								<button
									type="button"
									onClick={() => {
										onChange("");
										setIsOpen(false);
									}}
									className="block w-full rounded-sm px-3 py-2.5 text-left text-slate-500 text-sm transition-colors hover:bg-blue-50"
								>
									Wyczyść wybór
								</button>
							) : null}
							<div
								className="subtle-vertical-scroll overflow-y-auto"
								style={{ maxHeight: optionListMaxHeight }}
							>
								{visibleOptions.map((option) => {
									const isSelected = option.value === value;
									return (
										<button
											key={`${option.value}-${option.label}`}
											type="button"
											onClick={() => {
												onChange(option.value);
												setIsOpen(false);
											}}
											className={`block w-full rounded-sm px-3 py-2.5 text-left text-sm transition-colors ${
												isSelected
													? "bg-blue-100 text-blue-900"
													: "text-slate-800 hover:bg-blue-50 hover:text-blue-900"
											}`}
										>
											{option.label}
										</button>
									);
								})}
								{visibleOptions.length === 0 ? (
									<p className="px-3 py-2 text-slate-500 text-sm">{noResultsText}</p>
								) : null}
							</div>
						</div>,
						document.body,
					)
				: null}

			{invalid && errorMessage ? (
				<span className="mt-1 block text-rose-700 text-xs">{errorMessage}</span>
			) : null}
		</label>
	);
}
