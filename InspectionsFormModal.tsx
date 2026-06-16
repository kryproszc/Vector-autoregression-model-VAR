import { CalendarDays, Check, ChevronDown, ChevronsUpDown, Pencil } from "lucide-react";
import { DateCalendar } from "@mui/x-date-pickers/DateCalendar";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { AdapterDateFns } from "@mui/x-date-pickers/AdapterDateFns";
import {
	type Dispatch,
	type FormEvent,
	type SetStateAction,
	useEffect,
	useRef,
	useState,
} from "react";
import { createPortal } from "react-dom";
import { pl } from "date-fns/locale";

import {
	type AddInspectionForm,
	type InspectionPeopleOption,
	getUserDisplayName,
} from "@/features/inspections/components/inspections-panel.utils";
import { DateListEditor } from "@/shared/components/forms/DateListEditor";
import { NoLetterDateField } from "@/shared/components/forms/NoLetterDateField";
import { RegistryFormScaffold } from "@/shared/components/forms/RegistryFormScaffold";
import { SingleSelectPortalField } from "@/shared/components/forms/SingleSelectPortalField";

type InspectionNameVariant = "full" | "short" | "user";
type InspectionNameVariantColumnKey =
	| "nazwaPodmiotu"
	| "typInspekcji"
	| "zakresInspekcji"
	| "rynek"
	| "rodzajPodmiotu"
	| "status";
type InspectionNameVariantByColumn = Partial<
	Record<InspectionNameVariantColumnKey, InspectionNameVariant>
>;
type InspectionShortValuesByColumn = Partial<
	Record<InspectionNameVariantColumnKey, string>
>;

type InspectionsFormModalProps = {
	isOpen: boolean;
	isPreviewMode?: boolean;
	canStartEditFromPreview?: boolean;
	onStartEditFromPreview?: () => void;
	editingInspectionId: string | null;
	editingInspectionCode?: string | null;
	showRequiredFieldErrors?: boolean;
	addInspectionForm: AddInspectionForm;
	setAddInspectionForm: Dispatch<SetStateAction<AddInspectionForm>>;
	inspectionNameVariants?: InspectionNameVariantByColumn;
	previewShortValuesByColumn?: InspectionShortValuesByColumn;
	marketShortLabelByValue?: Record<string, string>;
	entityNameOptions: Array<{ value: string; label: string }>;
	inspectionTypeOptions: string[];
	inspectionScopeOptions: Array<{ value: string; label: string }>;
	marketOptions: string[];
	entityTypeOptions: string[];
	inspectionStatusOptions: Array<{ value: string; label: string }>;
	selectedInspectionScopes: string[];
	setSelectedInspectionScopes: Dispatch<SetStateAction<string[]>>;
	operatorDisplayName: string;
	operatorLogin: string;
	isTeamPickerOpen: boolean;
	setIsTeamPickerOpen: Dispatch<SetStateAction<boolean>>;
	selectedTeamMemberIds: number[];
	setSelectedTeamMemberIds: Dispatch<SetStateAction<number[]>>;
	selectedTeamMembers: string[];
	teamMemberScopeError?: string | null;
	outOfScopeTeamMemberUserId?: number | null;
	activeUsers: InspectionPeopleOption[];
	availableLeaderUsers: InspectionPeopleOption[];
	leaderChangeIrreversibleWarning?: string | null;
	forceLeaderSelectionReadonly?: boolean;
	selectedLeaderUserId: number | null;
	setSelectedLeaderUserId: Dispatch<SetStateAction<number | null>>;
	dataAkceptacjiNotyList: string[];
	setDataAkceptacjiNotyList: Dispatch<SetStateAction<string[]>>;
	isDataAkceptacjiNotyBrak: boolean;
	setIsDataAkceptacjiNotyBrak: Dispatch<SetStateAction<boolean>>;
	addInspectionError: string | null;
	isReadOnly?: boolean;
	isSaveDisabledDueToLock?: boolean;
	lockNotice?: string | null;
	onRetryAcquire?: () => void;
	inactivityIsWarning?: boolean;
	inactivitySecondsRemaining?: number;
	onInactivityContinue?: () => void;
	versionConflictUpdatedAt?: string | null;
	onRefreshAfterConflict?: () => void;
	isSubmittingInspection: boolean;
	onToggleTeamMember: (userId: number) => void;
	onClose: () => void;
	onSubmit: (event: FormEvent<HTMLFormElement>) => void;
};

function parseIsoDate(value: string): Date | undefined {
	if (!value) {
		return undefined;
	}

	const [yearText, monthText, dayText] = value.split("-");
	const year = Number(yearText);
	const month = Number(monthText);
	const day = Number(dayText);
	if (
		!Number.isInteger(year) ||
		!Number.isInteger(month) ||
		!Number.isInteger(day) ||
		month < 1 ||
		month > 12 ||
		day < 1 ||
		day > 31
	) {
		return undefined;
	}

	const date = new Date(year, month - 1, day);
	if (
		date.getFullYear() !== year ||
		date.getMonth() !== month - 1 ||
		date.getDate() !== day
	) {
		return undefined;
	}

	return date;
}

function toIsoDateValue(date: Date) {
	const year = date.getFullYear();
	const month = String(date.getMonth() + 1).padStart(2, "0");
	const day = String(date.getDate()).padStart(2, "0");
	return `${year}-${month}-${day}`;
}

const MIN_CALENDAR_DATE = new Date(2016, 0, 1);
const MAX_CALENDAR_DATE = new Date(2030, 11, 31);

const ENTITY_DISPLAY_SHORTCUTS: Array<[pattern: RegExp, shortLabel: string]> = [
	[/^Vienna Life Towarzystwo Ubezpieczeń na Życie Spółka Akcyjna Vienna Insurance Group$/i, "Vienna Life TU na Życie S.A. (VIG)"],
];

function getShortEntityDisplayLabel(value: string) {
	const normalized = value.trim();
	if (!normalized) {
		return "";
	}

	for (const [pattern, shortLabel] of ENTITY_DISPLAY_SHORTCUTS) {
		if (pattern.test(normalized)) {
			return shortLabel;
		}
	}

	return normalized;
}

function clampDateToCalendarRange(date: Date) {
	if (date < MIN_CALENDAR_DATE) {
		return MIN_CALENDAR_DATE;
	}

	if (date > MAX_CALENDAR_DATE) {
		return MAX_CALENDAR_DATE;
	}

	return date;
}

function formatDisplayDate(value: string) {
	const parsed = parseIsoDate(value);
	if (!parsed) {
		return "";
	}

	const day = String(parsed.getDate()).padStart(2, "0");
	const month = String(parsed.getMonth() + 1).padStart(2, "0");
	const year = parsed.getFullYear();
	return `${day}.${month}.${year}`;
}

function DateInputWithCalendar({
	label,
	value,
	onChange,
	disabled = false,
	labelClassName,
	invalid = false,
	minSelectableDate,
	maxSelectableDate,
	previewTextOnly = false,
	errorMessage = null,
}: {
	label: string;
	value: string;
	onChange: (next: string) => void;
	disabled?: boolean;
	labelClassName?: string;
	invalid?: boolean;
	minSelectableDate?: Date;
	maxSelectableDate?: Date;
	previewTextOnly?: boolean;
	errorMessage?: string | null;
}) {
	const [isCalendarOpen, setIsCalendarOpen] = useState(false);
	const [calendarView, setCalendarView] = useState<"year" | "month" | "day">(
		"day",
	);
	const containerRef = useRef<HTMLDivElement | null>(null);
	const popupRef = useRef<HTMLDivElement | null>(null);
	const [popupPosition, setPopupPosition] = useState<{
		top: number;
		left: number;
	} | null>(null);
	const effectiveMinDate = minSelectableDate
		? clampDateToCalendarRange(minSelectableDate)
		: MIN_CALENDAR_DATE;
	const effectiveMaxDate = maxSelectableDate
		? clampDateToCalendarRange(maxSelectableDate)
		: MAX_CALENDAR_DATE;
	const [tempDate, setTempDate] = useState<Date | null>(() =>
		parseIsoDate(value) ?? null,
	);
	const popupWidth = calendarView === "day" ? 288 : 336;
	const popupHeight = calendarView === "day" ? 420 : 372;
	const displayValue = formatDisplayDate(value) || "-";

	const updatePopupPosition = () => {
		const anchor = containerRef.current;
		if (!anchor) {
			return;
		}

		const anchorRect = anchor.getBoundingClientRect();
		const viewportPadding = 8;
		const offset = 8;
		const dialog = anchor.closest('[role="dialog"]') as HTMLElement | null;
		const dialogRect = dialog?.getBoundingClientRect() ?? null;
		const availableTop = Math.max(
			viewportPadding,
			dialogRect ? dialogRect.top + 8 : viewportPadding,
		);
		const availableBottom = Math.min(
			window.innerHeight,
			dialogRect ? dialogRect.bottom - 8 : window.innerHeight,
		);
		const spaceBelow = availableBottom - anchorRect.bottom;
		const shouldPreferUp = spaceBelow < popupHeight + offset;
		const requestedTop = shouldPreferUp
			? anchorRect.top - popupHeight - offset
			: anchorRect.bottom + offset;
		const maxTop = Math.max(availableTop, availableBottom - popupHeight);
		const top = Math.min(Math.max(requestedTop, availableTop), maxTop);
		const left = Math.min(
			Math.max(viewportPadding, anchorRect.right - popupWidth),
			window.innerWidth - popupWidth - viewportPadding,
		);

		setPopupPosition({ top, left });
	};

	useEffect(() => {
		setTempDate(parseIsoDate(value) ?? null);
	}, [value]);

	useEffect(() => {
		if (!isCalendarOpen) {
			setPopupPosition(null);
			return;
		}

		updatePopupPosition();
		const handleAnyScroll = (event: Event) => {
			const target = event.target as Node | null;
			if (target && popupRef.current?.contains(target)) {
				return;
			}
			setIsCalendarOpen(false);
		};

		window.addEventListener("resize", updatePopupPosition);
		window.addEventListener("scroll", handleAnyScroll, true);
		return () => {
			window.removeEventListener("resize", updatePopupPosition);
			window.removeEventListener("scroll", handleAnyScroll, true);
		};
	}, [isCalendarOpen, calendarView]);

	useEffect(() => {
		if (!isCalendarOpen) {
			return;
		}

		const handlePointerDown = (event: MouseEvent) => {
			const target = event.target as Node | null;
			if (!target) {
				return;
			}

			const isInsideAnchor =
				containerRef.current && containerRef.current.contains(target);
			const isInsidePopup = popupRef.current && popupRef.current.contains(target);

			if (!isInsideAnchor && !isInsidePopup) {
				setIsCalendarOpen(false);
			}
		};

		const handleEscape = (event: KeyboardEvent) => {
			if (event.key === "Escape") {
				setIsCalendarOpen(false);
			}
		};

		document.addEventListener("mousedown", handlePointerDown);
		document.addEventListener("keydown", handleEscape);

		return () => {
			document.removeEventListener("mousedown", handlePointerDown);
			document.removeEventListener("keydown", handleEscape);
		};
	}, [isCalendarOpen]);

	const handleClear = () => {
		onChange("");
		setTempDate(null);
		setCalendarView("day");
		setIsCalendarOpen(false);
	};

	const handleToday = () => {
		const today = clampDateToCalendarRange(new Date());
		const boundedToday =
			today < effectiveMinDate
				? effectiveMinDate
				: today > effectiveMaxDate
					? effectiveMaxDate
					: today;
		setTempDate(boundedToday);
		onChange(toIsoDateValue(boundedToday));
		setCalendarView("day");
		setIsCalendarOpen(false);
	};

	if (previewTextOnly) {
		return (
			<div className="text-slate-700 text-sm">
				<span className={`mb-1 block ${labelClassName ?? ""}`.trim()}>{label}</span>
				<p className="text-slate-900 text-sm font-semibold">{displayValue}</p>
			</div>
		);
	}

	return (
		<label className="text-slate-700 text-sm">
			<span className={`mb-1 block ${labelClassName ?? ""}`.trim()}>{label}</span>
			<div ref={containerRef} className="relative">
				<input
					type="text"
					value={formatDisplayDate(value)}
					placeholder="dd.mm.rrrr"
					readOnly
					disabled={disabled}
					onKeyDown={(event) => {
						if (disabled || !value) {
							return;
						}

						if (event.key === "Backspace" || event.key === "Delete") {
							event.preventDefault();
							handleClear();
						}
					}}
					onClick={() => {
						if (!disabled) {
							setIsCalendarOpen((prev) => {
								const next = !prev;
								if (next) {
									setTempDate(parseIsoDate(value) ?? null);
									setCalendarView("day");
								}
								return next;
							});
						}
					}}
					className={`w-full cursor-pointer rounded-lg border px-3 py-2 pr-10 text-sm outline-none transition-colors disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-700 ${
						invalid ? "border-rose-300 focus:border-rose-400" : "border-slate-300 focus:border-blue-400"
					}`}
				/>
				<button
					type="button"
					disabled={disabled}
					aria-label={`Otwórz kalendarz dla pola: ${label}`}
					onMouseDown={(event) => {
						event.preventDefault();
						event.stopPropagation();
					}}
					onClick={() => {
						if (disabled) {
							return;
						}

						setIsCalendarOpen((prev) => {
							const next = !prev;
							if (next) {
								setTempDate(parseIsoDate(value) ?? null);
								setCalendarView("day");
							}
							return next;
						});
					}}
					className="absolute top-1/2 right-2 inline-flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full border border-slate-200 bg-slate-50 text-slate-600 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
				>
					<CalendarDays size={13} />
				</button>

				{isCalendarOpen && !disabled && popupPosition
					? createPortal(
							<div
								ref={popupRef}
								className={`fixed z-[80] rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_16px_40px_rgba(15,23,42,0.14)] ${
									calendarView === "day" ? "w-[18rem]" : "w-[21rem]"
								}`}
								style={{
									top: popupPosition.top,
									left: popupPosition.left,
								}}
							>
								<LocalizationProvider dateAdapter={AdapterDateFns} adapterLocale={pl}>
									<DateCalendar
										value={tempDate}
										onChange={(nextValue) => {
											setTempDate(nextValue);
											if (calendarView === "day" && nextValue) {
												onChange(toIsoDateValue(nextValue));
												setCalendarView("day");
												setIsCalendarOpen(false);
											}
										}}
										view={calendarView}
										onViewChange={(nextView) => setCalendarView(nextView)}
										views={["year", "month", "day"]}
										openTo="day"
										minDate={effectiveMinDate}
										maxDate={effectiveMaxDate}
										referenceDate={tempDate ?? new Date()}
										sx={{
											width: "100%",
											maxHeight: calendarView === "day" ? 336 : 356,
											"& .MuiPickersCalendarHeader-root": {
												paddingLeft: 0,
												paddingRight: 0,
												marginBottom: "0.35rem",
											},
											"& .MuiPickersCalendarHeader-label": {
												fontSize: "1.05rem",
												fontWeight: 700,
												color: "#0f172a",
											},
											"& .MuiPickersArrowSwitcher-button": {
												color: "#64748b",
											},
											"& .MuiDayCalendar-weekDayLabel": {
												fontSize: "0.76rem",
												fontWeight: 600,
												color: "#64748b",
											},
											"& .MuiPickersDay-root": {
												fontSize: "0.95rem",
												fontWeight: 500,
												color: "#0f172a",
											},
											"& .MuiPickersDay-root.Mui-selected": {
												backgroundColor: "#1976d2",
												color: "#fff",
											},
											"& .MuiPickersDay-root.MuiPickersDay-today": {
												borderColor: "#94a3b8",
											},
											"& .MuiYearCalendar-button": {
												fontSize: "0.98rem",
												fontWeight: 500,
												color: "#0f172a",
											},
											"& .MuiYearCalendar-button.Mui-selected": {
												backgroundColor: "#1976d2",
												color: "#fff",
											},
											"& .MuiMonthCalendar-button": {
												fontSize: "0.98rem",
												fontWeight: 600,
												color: "#0f172a",
											},
											"& .MuiMonthCalendar-button.Mui-selected": {
												backgroundColor: "#1976d2",
												color: "#fff",
											},
											"& .MuiYearCalendar-root": {
												height: 252,
											},
											"& .MuiMonthCalendar-root": {
												height: 252,
											},
										}}
									/>
								</LocalizationProvider>

								<div className="mt-3 flex items-center justify-between border-slate-100 border-t pt-3">
									<button
										type="button"
										onClick={handleToday}
										className="font-bold text-[11px] text-slate-400 uppercase tracking-wide transition-colors hover:text-slate-500"
									>
										Dzisiaj
									</button>
									<button
										type="button"
										onClick={handleClear}
										className="rounded-lg bg-blue-50 px-4 py-2 font-bold text-[11px] text-blue-600 uppercase tracking-wide transition-colors hover:bg-blue-100"
									>
										Wyczyść
									</button>
								</div>
							</div>,
							document.body,
						)
					: null}
			</div>
			{errorMessage ? (
				<span className="mt-1 block text-rose-700 text-xs">{errorMessage}</span>
			) : null}
		</label>
	);
}

function MultiSelectPeoplePortalField({
	label,
	placeholder,
	options,
	values,
	onChange,
	disabled = false,
	selectedSummary,
	errorMessage = null,
	highlightedUserId = null,
}: {
	label: string;
	placeholder: string;
	options: Array<{
		id: number;
		label: string;
		teamId?: number | null;
		teamName?: string | null;
	}>;
	values: number[];
	onChange: (next: number[]) => void;
	disabled?: boolean;
	selectedSummary?: string;
	errorMessage?: string | null;
	highlightedUserId?: number | null;
}) {
	const [isOpen, setIsOpen] = useState(false);
	const [searchQuery, setSearchQuery] = useState("");
	const [selectedTeamIds, setSelectedTeamIds] = useState<number[]>([]);
	const triggerRef = useRef<HTMLButtonElement | null>(null);
	const popupRef = useRef<HTMLDivElement | null>(null);
	const [popupPosition, setPopupPosition] = useState<{
		top: number;
		left: number;
		width: number;
	} | null>(null);

	const selectedCount = values.length;
	const selectedSummaryItems = selectedSummary
		? selectedSummary
				.split(",")
				.map((item) => item.trim())
				.filter(Boolean)
		: [];
	const teamOptions = Array.from(
		options
			.reduce(
			(map, option) => {
				if (typeof option.teamId !== "number" || option.teamId <= 0) {
					return map;
				}

				const current = map.get(option.teamId);
				if (!current) {
					map.set(option.teamId, {
						id: option.teamId,
						label:
							typeof option.teamName === "string" && option.teamName.trim()
								? option.teamName.trim()
								: `Zespół ${option.teamId}`,
						memberCount: 1,
					});
					return map;
				}

				map.set(option.teamId, {
					...current,
					memberCount: current.memberCount + 1,
				});
				return map;
			},
			new Map<number, { id: number; label: string; memberCount: number }>(),
		)
			.values(),
	).sort((left, right) =>
		left.label.localeCompare(right.label, "pl", { sensitivity: "base" }),
	);
	const teamFilteredOptions =
		selectedTeamIds.length > 0
			? options.filter(
					(option) =>
						typeof option.teamId === "number" &&
						selectedTeamIds.includes(option.teamId),
				)
			: options;
	const canClearSelectedPeople = values.length > 0;
	const canResetTeamFilter = selectedTeamIds.length > 0;
	const normalizedSearchQuery = searchQuery.trim().toLowerCase();
	const visibleOptions = normalizedSearchQuery
		? teamFilteredOptions.filter((option) =>
				option.label.toLowerCase().includes(normalizedSearchQuery),
			)
		: teamFilteredOptions;

	const updatePopupPosition = () => {
		const trigger = triggerRef.current;
		if (!trigger) {
			return;
		}

		const rect = trigger.getBoundingClientRect();
		const viewportPadding = 8;
		const popupHeight = 280;
		const dialog = trigger.closest('[role="dialog"]') as HTMLElement | null;
		const dialogRect = dialog?.getBoundingClientRect() ?? null;
		const availableBottom = Math.min(
			window.innerHeight,
			dialogRect ? dialogRect.bottom - 8 : window.innerHeight,
		);
		const availableTop = Math.max(8, dialogRect ? dialogRect.top + 8 : 8);
		const spaceBelow = availableBottom - rect.bottom;
		const spaceAbove = rect.top - availableTop;
		const shouldOpenUp =
			spaceAbove >= popupHeight + 8 || spaceAbove > spaceBelow;

		setPopupPosition({
			top: shouldOpenUp ? rect.top - popupHeight - 8 : rect.bottom + 8,
			left: Math.min(
				Math.max(viewportPadding, rect.left),
				window.innerWidth - rect.width - viewportPadding,
			),
			width: rect.width,
		});
	};

	useEffect(() => {
		if (!isOpen) {
			setPopupPosition(null);
			setSearchQuery("");
			setSelectedTeamIds([]);
			return;
		}

		updatePopupPosition();
		const handleAnyScroll = () => {
			updatePopupPosition();
		};
		window.addEventListener("resize", updatePopupPosition);
		window.addEventListener("scroll", handleAnyScroll, true);
		return () => {
			window.removeEventListener("resize", updatePopupPosition);
			window.removeEventListener("scroll", handleAnyScroll, true);
		};
	}, [isOpen]);

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
			<div className="relative">
				<button
					ref={triggerRef}
					type="button"
					disabled={disabled}
					onClick={() => setIsOpen((prev) => !prev)}
					className={`flex w-full items-center justify-between rounded-lg border bg-white px-3 py-2 text-left text-sm transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-700 ${
						errorMessage ? "border-rose-300" : "border-slate-300"
					}`}
				>
					<span>
						{selectedCount > 0
							? `Wybrano osób: ${selectedCount}`
							: placeholder}
					</span>
					<ChevronDown size={14} className="text-slate-500" />
				</button>

				{isOpen && !disabled && popupPosition
					? createPortal(
							<div
								ref={popupRef}
								className="fixed z-[80] rounded-xl border border-slate-200 bg-white p-2.5 shadow-[0_14px_34px_rgba(15,23,42,0.14)]"
								style={{
									top: popupPosition.top,
									left: popupPosition.left,
									width: popupPosition.width,
								}}
							>
								<div className="mb-2 flex items-center justify-end gap-2 border-slate-200 border-b pb-2">
									<button
										type="button"
										disabled={!canClearSelectedPeople}
										onClick={() => onChange([])}
										className={`font-semibold text-xs transition-colors disabled:cursor-not-allowed ${
											canClearSelectedPeople
												? "text-blue-900 hover:text-blue-900"
												: "text-slate-400"
										}`}
									>
										Wyczyść
									</button>
									<button
										type="button"
										onClick={() => onChange(visibleOptions.map((option) => option.id))}
										className="font-semibold text-slate-600 text-xs hover:text-slate-900"
									>
										Zaznacz wszystkie
									</button>
								</div>

								{teamOptions.length > 0 ? (
									<div className="mb-2 rounded-md border border-slate-200 bg-slate-50/60 p-2">
										<div className="mb-1.5 flex items-center justify-between gap-2">
											<span className="font-medium text-[11px] text-slate-600 uppercase tracking-wide">
												Filtr zespołu
											</span>
											<button
												type="button"
												disabled={!canResetTeamFilter}
												onClick={() => setSelectedTeamIds([])}
												className={`font-semibold text-[11px] transition-colors disabled:cursor-not-allowed ${
													canResetTeamFilter
														? "text-blue-900 hover:text-blue-900"
														: "text-slate-400"
												}`}
											>
												Wszystkie zespoły
											</button>
										</div>
										<div className="flex flex-wrap gap-2">
											{teamOptions.map((teamOption) => {
												const isChecked = selectedTeamIds.includes(teamOption.id);

												return (
													<label
														key={`team-filter-${teamOption.id}`}
														className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-900 text-xs hover:bg-slate-100"
													>
														<input
															type="checkbox"
															checked={isChecked}
															className="h-3.5 w-3.5 rounded border-slate-300 text-blue-600 focus:ring-1 focus:ring-blue-300"
															onChange={(event) => {
																if (event.target.checked) {
																	setSelectedTeamIds((prev) => [...prev, teamOption.id]);
																	return;
																}

																setSelectedTeamIds((prev) =>
																	prev.filter((teamId) => teamId !== teamOption.id),
																);
															}}
														/>
														<span>{`${teamOption.label} (${teamOption.memberCount})`}</span>
													</label>
												);
											})}
										</div>
									</div>
								) : null}

								<div className="mb-2">
									<input
										type="text"
										value={searchQuery}
										onChange={(event) => setSearchQuery(event.target.value)}
										placeholder="Wyszukaj osobę..."
										className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 text-slate-900 text-sm outline-none focus:border-blue-400"
									/>
								</div>

								<div className="subtle-vertical-scroll max-h-52 space-y-1 overflow-y-auto pr-1 text-[13px]">
									{visibleOptions.map((option) => {
										const isSelected = values.includes(option.id);
										const isHighlighted = highlightedUserId === option.id;
										return (
											<label
												key={option.id}
												className={`flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 text-slate-900 hover:bg-slate-100 ${
													isHighlighted ? "bg-rose-50" : ""
												}`}
											>
												<input
													type="checkbox"
													checked={isSelected}
													className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-1 focus:ring-blue-300"
													onChange={(event) => {
														if (event.target.checked) {
															onChange([...values, option.id]);
															return;
														}

														onChange(values.filter((id) => id !== option.id));
													}}
												/>
												<span>
													{option.label}
													{isHighlighted ? " (poza zakresem)" : ""}
												</span>
											</label>
										);
									})}
									{visibleOptions.length === 0 ? (
										<p className="px-2 py-2 text-slate-500 text-sm">Brak wyników.</p>
									) : null}
								</div>
							</div>,
							document.body,
						)
					: null}
			</div>

			{errorMessage ? (
				<span className="mt-1 block text-rose-700 text-xs">{errorMessage}</span>
			) : null}
		</label>
	);
}

function PreviewTeamMembersField({
	label,
	members,
}: {
	label: string;
	members: string[];
}) {
	return (
		<div className="max-w-[32rem] text-slate-700 text-sm">
			<span className="mb-1 block">{label}</span>
			{members.length > 0 ? (
				<div className="subtle-vertical-scroll mt-1 max-h-[6.5rem] space-y-0.5 overflow-y-auto pr-1 text-slate-900 text-sm font-semibold">
					{members.map((member, index) => (
						<div key={`preview-team-member-${member}-${index}`}>
							{index + 1}. {member}
						</div>
					))}
				</div>
			) : (
				<p className="mt-1 text-slate-500 text-sm font-semibold">Brak osób w składzie.</p>
			)}
		</div>
	);
}

function PreviewInspectionScopesField({
	label,
	scopes,
}: {
	label: string;
	scopes: string[];
}) {
	return (
		<div className="max-w-[32rem] text-slate-700 text-sm">
			<span className="mb-1 block">{label}</span>
			{scopes.length > 0 ? (
				<div className="subtle-vertical-scroll mt-1 max-h-[6.5rem] space-y-0.5 overflow-y-auto break-words pr-1 text-slate-900 text-sm font-semibold">
					{scopes.map((scope, index) => (
						<div key={`preview-scope-${scope}-${index}`}>
							{index + 1}. {scope}
						</div>
					))}
				</div>
			) : null}
		</div>
	);
}

export function InspectionsFormModal({
	isOpen,
	isPreviewMode = false,
	canStartEditFromPreview = false,
	onStartEditFromPreview,
	editingInspectionId,
	editingInspectionCode,
	showRequiredFieldErrors = false,
	addInspectionForm,
	setAddInspectionForm,
	entityNameOptions,
	inspectionTypeOptions,
	inspectionScopeOptions,
	marketOptions,
	entityTypeOptions,
	inspectionStatusOptions,
	selectedInspectionScopes,
	setSelectedInspectionScopes,
	operatorDisplayName,
	operatorLogin,
	isTeamPickerOpen,
	setIsTeamPickerOpen,
	selectedTeamMemberIds,
	setSelectedTeamMemberIds,
	selectedTeamMembers,
	teamMemberScopeError = null,
	outOfScopeTeamMemberUserId = null,
	activeUsers,
	availableLeaderUsers,
	leaderChangeIrreversibleWarning = null,
	forceLeaderSelectionReadonly = false,
	selectedLeaderUserId,
	setSelectedLeaderUserId,
	dataAkceptacjiNotyList,
	setDataAkceptacjiNotyList,
	isDataAkceptacjiNotyBrak,
	setIsDataAkceptacjiNotyBrak,
	addInspectionError,
	isReadOnly = false,
	isSaveDisabledDueToLock = false,
	lockNotice = null,
	onRetryAcquire,
	inactivityIsWarning = false,
	inactivitySecondsRemaining = 0,
	onInactivityContinue,
	versionConflictUpdatedAt = null,
	onRefreshAfterConflict,
	isSubmittingInspection,
	onToggleTeamMember,
	onClose,
	onSubmit,
}: InspectionsFormModalProps) {
	const [isLeaderPickerOpen, setIsLeaderPickerOpen] = useState(false);
	const [isScopePickerOpen, setIsScopePickerOpen] = useState(false);
	const [scopeSearchQuery, setScopeSearchQuery] = useState("");
	const [leaderSearchQuery, setLeaderSearchQuery] = useState("");
	const scopePickerRef = useRef<HTMLDivElement | null>(null);
	const leaderPickerRef = useRef<HTMLDivElement | null>(null);
	const teamPickerRef = useRef<HTMLDivElement | null>(null);

	useEffect(() => {
		if (!isScopePickerOpen) {
			setScopeSearchQuery("");
			return;
		}

		const handlePointerDown = (event: MouseEvent) => {
			const target = event.target as Node | null;
			if (!target) {
				return;
			}

			if (scopePickerRef.current && !scopePickerRef.current.contains(target)) {
				setIsScopePickerOpen(false);
			}
		};

		document.addEventListener("mousedown", handlePointerDown);
		return () => {
			document.removeEventListener("mousedown", handlePointerDown);
		};
	}, [isScopePickerOpen]);

	useEffect(() => {
		if (!isLeaderPickerOpen) {
			setLeaderSearchQuery("");
			return;
		}

		const handlePointerDown = (event: MouseEvent) => {
			const target = event.target as Node | null;
			if (!target) {
				return;
			}

			if (leaderPickerRef.current && !leaderPickerRef.current.contains(target)) {
				setIsLeaderPickerOpen(false);
			}
		};

		document.addEventListener("mousedown", handlePointerDown);
		return () => {
			document.removeEventListener("mousedown", handlePointerDown);
		};
	}, [isLeaderPickerOpen]);

	useEffect(() => {
		if (!isTeamPickerOpen) {
			return;
		}

		const handlePointerDown = (event: MouseEvent) => {
			const target = event.target as Node | null;
			if (!target) {
				return;
			}

			if (teamPickerRef.current && !teamPickerRef.current.contains(target)) {
				setIsTeamPickerOpen(false);
			}
		};

		document.addEventListener("mousedown", handlePointerDown);
		return () => {
			document.removeEventListener("mousedown", handlePointerDown);
		};
	}, [isTeamPickerOpen, setIsTeamPickerOpen]);

	if (!isOpen) {
		return null;
	}

	const selectedLeaderLabel = availableLeaderUsers.find(
		(user) => user.id === selectedLeaderUserId,
	)
		? getUserDisplayName(
				availableLeaderUsers.find((user) => user.id === selectedLeaderUserId)!,
			)
		: "Wybierz osobę kierującą";
	const isLeaderSelectionLocked =
		forceLeaderSelectionReadonly || availableLeaderUsers.length <= 1;
	const normalizedScopeSearchQuery = scopeSearchQuery.trim().toLowerCase();
	const filteredInspectionScopeOptions = normalizedScopeSearchQuery
		? inspectionScopeOptions.filter((option) =>
				option.label.toLowerCase().includes(normalizedScopeSearchQuery),
			)
		: inspectionScopeOptions;
	const entityNameDisplayOptions = entityNameOptions.map((option) => ({
		...option,
		label: getShortEntityDisplayLabel(option.label),
	}));
	const normalizedLeaderSearchQuery = leaderSearchQuery.trim().toLowerCase();
	const filteredLeaderUsers = normalizedLeaderSearchQuery
		? availableLeaderUsers.filter((user) =>
				getUserDisplayName(user)
					.toLowerCase()
					.includes(normalizedLeaderSearchQuery),
			)
		: availableLeaderUsers;
	const selectedInspectionScopeLabels = selectedInspectionScopes.map((scopeValue) => {
		const matchedOption = inspectionScopeOptions.find(
			(option) => option.value === scopeValue,
		);
		return matchedOption?.label ?? scopeValue;
	});
	const normalizedInspectionType = addInspectionForm.typInspekcji
		.trim()
		.toLowerCase();
	const hasInspectionTypeSelected = Boolean(normalizedInspectionType);
	const isControlType = normalizedInspectionType.includes("kontrol");
	const isSupervisoryVisitType = normalizedInspectionType.includes("wizyta");
	const leaderFieldLabel =
		isControlType && !isSupervisoryVisitType
			? "Osoba kierująca kontrolą *"
			: isSupervisoryVisitType && !isControlType
				? "Osoba kierująca wizytą nadzorczą *"
				: "Osoba kierująca kontrolą/wizytą *";
	const protocolOrReportLabel =
		isControlType && !isSupervisoryVisitType
			? "Data protokołu kontroli"
			: isSupervisoryVisitType && !isControlType
				? "Data sprawozdania z wizyty nadzorczej"
				: "Data protokołu / sprawozdania z wizyty nadzorczej";
	const protocolDeliveryLabel =
		isControlType && !isSupervisoryVisitType
			? "Data doręczenia protokołu kontroli"
			: "Data doręczenia protokołu kontroli";
	const reportAcceptanceLabel =
		isSupervisoryVisitType && !isControlType
			? "Data akceptacji sprawozdania z wizyty nadzorczej"
			: "Data akceptacji sprawozdania z wizyty nadzorczej";
	const visitLetterDeliveryLabel = isSupervisoryVisitType
		? "Data doręczenia pisma do podmiotu z ustaleniami wizyty nadzorczej"
		: "Data doręczenia pisma do podmiotu z ustaleniami wizyty nadzorczej";
	const objectionsLetterLabel =
		isControlType && !isSupervisoryVisitType
			? "Data pisma podmiotu z zastrzeżeniami do protokołu kontroli"
			: isSupervisoryVisitType && !isControlType
				? "Data uwag do pisma po wizycie nadzorczej"
				: "Data pisma podmiotu z zastrzeżeniami do protokołu kontroli / uwagami do pisma po wizycie nadzorczej";
	const objectionsReceivedLabel =
		isControlType && !isSupervisoryVisitType
			? "Data wpływu pisma podmiotu z zastrzeżeniami do protokołu kontroli"
			: isSupervisoryVisitType && !isControlType
				? "Data wpływu uwag do pisma po wizycie nadzorczej"
				: "Data wpływu pisma podmiotu z zastrzeżeniami do protokołu kontroli / uwagami do pisma po wizycie nadzorczej";
	const objectionsLetterSentLabel =
		isControlType && !isSupervisoryVisitType
			? "Data wysłania pisma podmiotu z zastrzeżeniami do protokołu kontroli"
			: isSupervisoryVisitType && !isControlType
				? "Data wysłania uwag do pisma po wizycie nadzorczej"
				: "Data wysłania pisma podmiotu z zastrzeżeniami do protokołu kontroli / uwagami do pisma po wizycie nadzorczej";
	const controlResponseLetterLabel = isControlType
		? "Data pisma z odpowiedzią na zastrzeżenia do protokołu kontroli"
		: "Data pisma z odpowiedzią na zastrzeżenia do protokołu kontroli";
	const controlResponseLetterSentLabel =
		isControlType && !isSupervisoryVisitType
			? "Data wysłania pisma z odpowiedzią na zastrzeżenia do protokołu kontroli"
			: isSupervisoryVisitType && !isControlType
				? "Data wysłania pisma z odpowiedzią na uwagi do pisma po wizycie nadzorczej"
				: "Data wysłania pisma z odpowiedzią na zastrzeżenia do protokołu kontroli / uwagi do pisma po wizycie nadzorczej";
	const isAspektKonsumenckiChecked =
		addInspectionForm.aspektKonsumencki.trim().toUpperCase() === "TAK";
	const stableFieldLabelClassName = "min-h-8 leading-5";
	const parsedRecommendationDates = addInspectionForm.dataZalecen
		.split(",")
		.map((value) => value.trim())
		.filter(Boolean)
		.map((value) => formatDisplayDate(value) || value);
	const parsedAcceptanceDates = dataAkceptacjiNotyList
		.map((value) => formatDisplayDate(value) || value)
		.filter(Boolean);
	const isRequiredInspectionTypeMissing =
		showRequiredFieldErrors && !addInspectionForm.typInspekcji.trim();
	const isRequiredEntityNameMissing =
		showRequiredFieldErrors && !addInspectionForm.nazwaPodmiotu.trim();
	const isRequiredStartDateMissing =
		showRequiredFieldErrors && !addInspectionForm.poczatekInspekcji;
	const isRequiredEndDateMissing =
		showRequiredFieldErrors && !addInspectionForm.koniecInspekcji;
	const isRequiredStatusMissing = showRequiredFieldErrors && !addInspectionForm.status.trim();
	const inspectionStartDate = parseIsoDate(addInspectionForm.poczatekInspekcji);
	const inspectionEndDate = parseIsoDate(addInspectionForm.koniecInspekcji);

	const renderPreviewTextField = (label: string, value: string, emptyLabel = "-") => {
		if (!isPreviewMode) {
			return null;
		}

		const normalizedValue = value.trim();
		return (
			<div className="text-slate-700 text-sm">
				<span className="mb-1 block">{label}</span>
				<p className="text-slate-900 text-sm font-semibold">
					{normalizedValue || emptyLabel}
				</p>
			</div>
		);
	};

	const renderNoLetterField = ({
		label,
		value,
		isNoLetter,
		disabled = false,
		onChangeValue,
		onChangeNoLetter,
	}: {
		label: string;
		value: string;
		isNoLetter: boolean;
		disabled?: boolean;
		onChangeValue: (nextValue: string) => void;
		onChangeNoLetter: (nextNoLetter: boolean) => void;
	}) => {
		if (isPreviewMode) {
			return (
				<div className="text-slate-700 text-sm">
					<label className={`mb-1 block text-slate-600 text-sm ${stableFieldLabelClassName}`}>
						{label}
					</label>
					<p className="text-slate-900 text-sm font-semibold">
						{isNoLetter ? "Brak pisma" : formatDisplayDate(value) || "-"}
					</p>
				</div>
			);
		}

		return (
			<NoLetterDateField
				label={label}
				labelClassName={stableFieldLabelClassName}
				value={value}
				isNoLetter={isNoLetter}
				disabled={disabled}
				onChangeValue={onChangeValue}
				onChangeNoLetter={onChangeNoLetter}
			/>
		);
	};

	const handleFormSubmit = (event: FormEvent<HTMLFormElement>) => {
		if (isPreviewMode) {
			event.preventDefault();
			onStartEditFromPreview?.();
			return;
		}

		onSubmit(event);
	};

	return (
		<RegistryFormScaffold
			isOpen={isOpen}
			title={
				isPreviewMode
					? "Podgląd inspekcji"
					: editingInspectionId
						? "Edytuj inspekcję"
						: "Dodaj inspekcję"
			}
			subtitle={
				editingInspectionId && editingInspectionCode
					? `Id inspekcji: ${editingInspectionCode}`
					: undefined
			}
			onClose={onClose}
			onSubmit={handleFormSubmit}
			closeOnBackdropClick={false}
			maxWidthClassName="max-w-[1900px]"
			isContentReadOnly={isReadOnly && !isPreviewMode}
			cancelLabel={isPreviewMode ? null : editingInspectionId ? "Anuluj" : undefined}
			headerNotices={
				<>
					{inactivityIsWarning ? (
						<div className="mt-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-amber-900 text-sm">
							<p className="font-semibold">
								Nie wykryto aktywności. Formularz zostanie zamknięty za{" "}
								<span className="tabular-nums">{inactivitySecondsRemaining}</span> s.
							</p>
							{onInactivityContinue ? (
								<button
									type="button"
									onClick={onInactivityContinue}
									className="mt-2 inline-flex h-7 items-center rounded border border-amber-400 bg-amber-100 px-2 font-semibold text-amber-900 text-xs transition-colors hover:bg-amber-200"
								>
									Kontynuuj edycję
								</button>
							) : null}
						</div>
					) : null}
					{lockNotice ? (
						<div className="mt-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-rose-800 text-sm">
							<p className="font-medium">{lockNotice}</p>
							{onRetryAcquire ? (
								<button
									type="button"
									onClick={onRetryAcquire}
									className="mt-1 inline-flex items-center font-medium text-xs underline underline-offset-2 hover:no-underline"
								>
									Spróbuj ponownie
								</button>
							) : null}
						</div>
					) : null}
				</>
			}
			footerContent={
				<>
					{addInspectionError ? (
						<div className="mb-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-rose-700 text-sm">
							<p className="font-medium">{addInspectionError}</p>
							{versionConflictUpdatedAt ? (
								<p className="mt-1 text-rose-700/90">
									Aktualna wersja rekordu: {versionConflictUpdatedAt}
								</p>
							) : null}
						</div>
					) : null}

					{versionConflictUpdatedAt && onRefreshAfterConflict ? (
						<div className="mb-2">
							<button
								type="button"
								onClick={onRefreshAfterConflict}
								className="inline-flex h-8 items-center rounded-md border border-amber-300 bg-amber-50 px-3 font-semibold text-amber-800 text-xs transition-colors hover:bg-amber-100"
							>
								Odśwież dane
							</button>
						</div>
					) : null}
				</>
			}
			isSubmitDisabled={
				isPreviewMode
					? !canStartEditFromPreview
					: isSubmittingInspection || isReadOnly || isSaveDisabledDueToLock
			}
			submitButtonClassName={
				isPreviewMode
					? "inline-flex h-10 items-center gap-2 rounded-lg border px-3.5 font-semibold text-sm transition-colors enabled:border-[#93b9ee] enabled:bg-[#d9e9ff] enabled:text-[#21508f] enabled:hover:bg-[#c9e0ff] disabled:cursor-not-allowed disabled:border-slate-300 disabled:bg-slate-200 disabled:text-slate-500"
					: undefined
			}
			submitLabel={
				isSubmittingInspection
					? "Zapisywanie..."
					: isPreviewMode
						? (
							<span className="inline-flex items-center gap-2">
								<Pencil size={15} />
								Edytuj
							</span>
						)
						: isReadOnly
						? "Tylko podgląd"
						: editingInspectionId
							? "Zapisz"
							: "Dodaj"
			}
		>
			<div className="space-y-4">
					<fieldset
						disabled={isReadOnly}
						className={`border-0 p-0 ${
							isPreviewMode
								? "[&_input:disabled]:!bg-white [&_input:disabled]:!text-slate-700 [&_textarea:disabled]:!bg-white [&_textarea:disabled]:!text-slate-700 [&_button:disabled]:!bg-white [&_button:disabled]:!text-slate-700 [&_select:disabled]:!bg-white [&_select:disabled]:!text-slate-700"
								: ""
						}`}
					>
						<div className="grid gap-4 xl:grid-cols-12 xl:items-start">
						<section className="rounded-xl border border-slate-200 p-3 xl:col-span-5">
							<h4 className="mb-3 font-semibold text-slate-700 text-sm uppercase tracking-wide">
								Dane podstawowe
							</h4>

							<div className="grid gap-3 xl:grid-cols-2">
								{isPreviewMode ? (
									renderPreviewTextField("Typ inspekcji *", addInspectionForm.typInspekcji)
								) : (
									<SingleSelectPortalField
										label="Typ inspekcji *"
										value={addInspectionForm.typInspekcji}
										options={inspectionTypeOptions}
										placeholder="• Wybierz typ inspekcji"
										invalid={isRequiredInspectionTypeMissing}
										errorMessage={
											isRequiredInspectionTypeMissing ? "Pole wymagane." : null
										}
										onChange={(next) =>
											setAddInspectionForm((prev) => ({
												...prev,
												typInspekcji: next,
												dataDoreczeniaProtokolu: next
													.toLowerCase()
													.includes("kontrol")
													? prev.dataDoreczeniaProtokolu
													: "",
												dataPismaZOdpowiedzia: next
													.toLowerCase()
													.includes("kontrol")
													? prev.dataPismaZOdpowiedzia
													: "",
												brakDataPismaZOdpowiedzia: next
													.toLowerCase()
													.includes("kontrol")
													? prev.brakDataPismaZOdpowiedzia
													: false,
												dataWyslaniaPismaZOdpowiedzia: next
													.toLowerCase()
													.includes("kontrol")
													? prev.dataWyslaniaPismaZOdpowiedzia
													: "",
												brakDataWyslaniaPismaZOdpowiedzia: next
													.toLowerCase()
													.includes("kontrol")
													? prev.brakDataWyslaniaPismaZOdpowiedzia
													: false,
												dataAkceptacjiSprawozdania: next
													.toLowerCase()
													.includes("wizyta")
													? prev.dataAkceptacjiSprawozdania
													: "",
												dataDoreczeniaPisma: next
													.toLowerCase()
													.includes("wizyta")
													? prev.dataDoreczeniaPisma
													: "",
												brakDataDoreczeniaPisma: next
													.toLowerCase()
													.includes("wizyta")
													? prev.brakDataDoreczeniaPisma
													: false,
											}))
										}
										disabled={isReadOnly || Boolean(editingInspectionId)}
									/>
								)}

								{isPreviewMode ? (
									renderPreviewTextField(
										"Nazwa podmiotu *",
										getShortEntityDisplayLabel(addInspectionForm.nazwaPodmiotu),
									)
								) : (
									<SingleSelectPortalField
										label="Nazwa podmiotu *"
										value={addInspectionForm.nazwaPodmiotu}
										options={entityNameDisplayOptions}
										placeholder="• Wybierz podmiot"
										enableSearch
										searchPlaceholder="Wyszukaj podmiot..."
										invalid={isRequiredEntityNameMissing}
										errorMessage={
											isRequiredEntityNameMissing ? "Pole wymagane." : null
										}
										onChange={(next) =>
											setAddInspectionForm((prev) => ({
												...prev,
												nazwaPodmiotu: next,
											}))
										}
										disabled={isReadOnly}
									/>
								)}

								{isPreviewMode ? (
									<PreviewInspectionScopesField
										label="Zakres inspekcji według upoważnienia"
										scopes={selectedInspectionScopeLabels}
									/>
								) : (
									<label className="text-slate-700 text-sm">
									<span className="mb-1 block">Zakres inspekcji według upoważnienia</span>
									<div ref={scopePickerRef} className="relative">
										<button
											type="button"
											onClick={() => setIsScopePickerOpen((prev) => !prev)}
											className="flex w-full items-center justify-between rounded-lg border border-slate-300 bg-white px-3 py-2 text-left text-slate-900 text-sm transition-colors hover:bg-slate-50"
										>
											<span>
												{selectedInspectionScopes.length
													? `Wybrano pozycji: ${selectedInspectionScopes.length}`
													: "Wybierz zakresy inspekcji"}
											</span>
											<ChevronsUpDown size={14} className="text-slate-500" />
										</button>

										{isScopePickerOpen ? (
											<div className="absolute z-20 mt-2 w-full rounded-lg border border-slate-300 bg-white p-2 shadow-lg">
												<div className="mb-2 flex items-center justify-end gap-2 border-slate-200 border-b pb-2">
													<button
														type="button"
														onClick={() => setSelectedInspectionScopes([])}
														className="font-semibold text-slate-600 text-xs hover:text-slate-900"
													>
														Wyczysc
													</button>
													<button
														type="button"
														onClick={() =>
															setSelectedInspectionScopes([
																...inspectionScopeOptions.map((option) => option.value),
															])
														}
														className="font-semibold text-slate-600 text-xs hover:text-slate-900"
													>
														Zaznacz wszystkie
													</button>
												</div>

													<div className="mb-2">
														<input
															type="text"
															value={scopeSearchQuery}
															onChange={(event) => setScopeSearchQuery(event.target.value)}
															placeholder="Wyszukaj zakres..."
															className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 text-slate-900 text-sm outline-none focus:border-blue-400"
														/>
													</div>

													<div className="subtle-vertical-scroll max-h-44 space-y-1 overflow-y-auto pr-1">
													{filteredInspectionScopeOptions.map((option) => {
														const isSelected =
															selectedInspectionScopes.includes(option.value);

														return (
															<button
																key={option.value}
																type="button"
																onClick={() => {
																	setSelectedInspectionScopes((prev) =>
																		prev.includes(option.value)
																			? prev.filter((item) => item !== option.value)
																			: [...prev, option.value],
																	);
																}}
																className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-slate-900 text-sm transition-colors hover:bg-slate-100"
															>
																<span
																	className={`flex h-4 w-4 items-center justify-center rounded border ${
																		isSelected
																			? "border-blue-600 bg-blue-600 text-white"
																			: "border-slate-300 bg-white"
																	}`}
																>
																	{isSelected ? <Check size={12} /> : null}
																</span>
																<span>{option.label}</span>
															</button>
														);
													})}
													{filteredInspectionScopeOptions.length === 0 ? (
														<p className="px-2 py-2 text-slate-500 text-sm">Brak wyników.</p>
													) : null}
												</div>
											</div>
										) : null}
									</div>
									</label>
									)}

								{isPreviewMode ? (
									renderPreviewTextField(
										"Czy dotyczy aspektu konsumenckiego?",
										isAspektKonsumenckiChecked ? "Tak" : "Nie",
									)
								) : (
									<label className="text-slate-700 text-sm">
										<span className="mb-1 block">Czy dotyczy aspektu konsumenckiego?</span>
										<div className="flex min-h-10 items-center rounded-lg border border-slate-300 bg-white px-3 py-2">
											<label className="inline-flex cursor-pointer items-center gap-2 text-slate-900 text-sm">
												<input
													type="checkbox"
													checked={isAspektKonsumenckiChecked}
													onChange={(event) =>
														setAddInspectionForm((prev) => ({
															...prev,
															aspektKonsumencki: event.target.checked ? "TAK" : "NIE",
														}))
													}
													className="h-4 w-4"
												/>
												<span>Tak</span>
											</label>
										</div>
									</label>
								)}

										{isPreviewMode ? (
											renderPreviewTextField(
												"Szczegóły dotyczące zakresu",
												addInspectionForm.szczegolyDotyczaceZakresu,
											)
										) : (
											<label className="text-slate-700 text-sm sm:col-span-2">
												<span className="mb-1 block">Szczegóły dotyczące zakresu</span>
												<textarea
													rows={2}
													value={addInspectionForm.szczegolyDotyczaceZakresu}
													onChange={(event) =>
														setAddInspectionForm((prev) => ({
															...prev,
															szczegolyDotyczaceZakresu: event.target.value,
														}))
													}
													className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 text-sm outline-none transition-colors focus:border-blue-400"
												/>
											</label>
										)}

								<DateInputWithCalendar
									label="Początek inspekcji *"
									value={addInspectionForm.poczatekInspekcji}
									previewTextOnly={isPreviewMode}
									maxSelectableDate={inspectionEndDate}
									invalid={isRequiredStartDateMissing}
									errorMessage={
										isRequiredStartDateMissing ? "Pole wymagane." : null
									}
									onChange={(nextStartDate) =>
										setAddInspectionForm((prev) => {
											const shouldAdjustEndDate =
												Boolean(prev.koniecInspekcji) &&
												Boolean(nextStartDate) &&
												prev.koniecInspekcji < nextStartDate;

											return {
												...prev,
												poczatekInspekcji: nextStartDate,
												koniecInspekcji: shouldAdjustEndDate
													? nextStartDate
													: prev.koniecInspekcji,
											};
										})
									}
								/>

								<DateInputWithCalendar
									label="Koniec inspekcji *"
									value={addInspectionForm.koniecInspekcji}
									previewTextOnly={isPreviewMode}
									minSelectableDate={inspectionStartDate}
									invalid={isRequiredEndDateMissing}
									errorMessage={
										isRequiredEndDateMissing ? "Pole wymagane." : null
									}
									onChange={(next) =>
										setAddInspectionForm((prev) => ({
											...prev,
											koniecInspekcji: next,
										}))
									}
								/>

								<label className="text-slate-700 text-sm">
									<span className="mb-1 block">
										{leaderFieldLabel}
									</span>
									{isPreviewMode ? (
										<p className="text-slate-900 text-sm font-semibold">
											{selectedLeaderLabel}
										</p>
									) : isLeaderSelectionLocked ? (
										<div className="flex w-full items-center rounded-lg border border-slate-300 bg-slate-100 px-3 py-2 text-slate-600 text-sm">
											{selectedLeaderLabel}
										</div>
									) : (
										<div ref={leaderPickerRef} className="relative">
											<button
												type="button"
												onClick={() => setIsLeaderPickerOpen((prev) => !prev)}
												className="flex w-full items-center justify-between rounded-lg border border-slate-300 bg-white px-3 py-2 text-left text-slate-900 text-sm transition-colors hover:bg-slate-50"
											>
												<span>{selectedLeaderLabel}</span>
												<ChevronsUpDown size={14} className="text-slate-500" />
											</button>

											{isLeaderPickerOpen ? (
												<div className="absolute z-20 mt-2 w-full rounded-lg border border-slate-300 bg-white p-2 shadow-lg">
													<div className="mb-2">
														<input
															type="text"
															value={leaderSearchQuery}
															onChange={(event) => setLeaderSearchQuery(event.target.value)}
															placeholder="Wyszukaj osobę..."
															className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 text-slate-900 text-sm outline-none focus:border-blue-400"
														/>
													</div>
													{availableLeaderUsers.length === 0 ? (
														<p className="px-2 py-1 text-slate-500 text-sm">
															Brak dostępnych użytkowników.
														</p>
													) : filteredLeaderUsers.length === 0 ? (
														<p className="px-2 py-1 text-slate-500 text-sm">Brak wyników.</p>
													) : (
														<div className="subtle-vertical-scroll max-h-44 space-y-1 overflow-y-auto pr-1">
															{filteredLeaderUsers.map((user) => {
																const optionLabel = getUserDisplayName(user);
																const isSelected =
																	user.id === selectedLeaderUserId;

																return (
																	<button
																		key={user.id}
																		type="button"
																		onClick={() => {
																			setSelectedLeaderUserId(user.id);
																			setIsLeaderPickerOpen(false);
																		}}
																		className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-slate-900 text-sm transition-colors hover:bg-slate-100"
																	>
																		<span
																			className={`flex h-4 w-4 items-center justify-center rounded border ${
																				isSelected
																					? "border-blue-600 bg-blue-600 text-white"
																					: "border-slate-300 bg-white"
																			}`}
																		>
																			{isSelected ? <Check size={12} /> : null}
																		</span>
																		<span>{optionLabel}</span>
																	</button>
																);
															})}
														</div>
													)}
												</div>
											) : null}
										</div>
									)}
								</label>

								{leaderChangeIrreversibleWarning ? (
									<div className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-amber-900 text-xs leading-5 xl:col-span-2">
										{leaderChangeIrreversibleWarning}
									</div>
								) : null}

								{isPreviewMode ? (
									renderPreviewTextField("Rynek", addInspectionForm.rynek)
								) : (
									<SingleSelectPortalField
										label="Rynek"
										value={addInspectionForm.rynek}
										options={marketOptions}
										placeholder="Wybierz rynek"
										onChange={(next) =>
											setAddInspectionForm((prev) => ({
												...prev,
												rynek: next,
											}))
										}
										disabled={isReadOnly}
									/>
								)}

								<div className="sm:col-span-2">
									{isPreviewMode ? (
										<PreviewTeamMembersField
											label="Skład zespołu"
											members={selectedTeamMembers}
										/>
									) : (
										<MultiSelectPeoplePortalField
											label="Skład zespołu"
											placeholder="Wybierz członków zespołu"
											options={activeUsers.map((user) => ({
												id: user.id,
												label: getUserDisplayName(user),
												teamId: user.teamId,
												teamName: user.teamName,
											}))}
											values={selectedTeamMemberIds}
											onChange={setSelectedTeamMemberIds}
											selectedSummary={selectedTeamMembers.join(", ")}
											errorMessage={teamMemberScopeError}
											highlightedUserId={outOfScopeTeamMemberUserId}
										/>
									)}
								</div>

								{isPreviewMode ? (
									renderPreviewTextField("Rodzaj podmiotu", addInspectionForm.rodzajPodmiotu)
								) : (
									<SingleSelectPortalField
										label="Rodzaj podmiotu"
										value={addInspectionForm.rodzajPodmiotu}
										options={entityTypeOptions}
										placeholder="Wybierz rodzaj podmiotu"
										onChange={(next) =>
											setAddInspectionForm((prev) => ({
												...prev,
												rodzajPodmiotu: next,
											}))
										}
										disabled={isReadOnly}
									/>
								)}

								<div>
									{isPreviewMode ? (
										renderPreviewTextField("Status *", addInspectionForm.status)
									) : (
										<SingleSelectPortalField
											label="Status *"
											value={addInspectionForm.status}
											options={inspectionStatusOptions}
											placeholder="Wybierz status"
											invalid={isRequiredStatusMissing}
											errorMessage={
												isRequiredStatusMissing ? "Pole wymagane." : null
											}
											onChange={(next) =>
												setAddInspectionForm((prev) => ({
													...prev,
													status: next,
												}))
											}
											disabled={isReadOnly}
										/>
									)}
								</div>

								<div className="sm:col-span-2">
									{isPreviewMode ? (
										renderPreviewTextField("Komentarz", addInspectionForm.komentarz)
									) : (
										<label className="text-slate-700 text-sm">
											<span className="mb-1 block">Komentarz</span>
											<textarea
												rows={2}
												value={addInspectionForm.komentarz}
												onChange={(event) =>
													setAddInspectionForm((prev) => ({
														...prev,
														komentarz: event.target.value,
													}))
												}
												className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 text-sm outline-none transition-colors focus:border-blue-400"
											/>
										</label>
									)}
								</div>


							</div>

						</section>

						<section className="rounded-xl border border-slate-200 p-3 xl:col-span-7">
							<h4 className="mb-3 font-semibold text-slate-700 text-sm uppercase tracking-wide">
								Terminy
							</h4>

							{!hasInspectionTypeSelected ? (
								<div className="flex min-h-[480px] items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50/60 px-6 text-center">
									<p className="font-medium text-slate-600 text-sm">
										Wybierz Typ inspekcji
									</p>
								</div>
							) : (
							<div className="grid gap-3 xl:grid-cols-2">
								<DateInputWithCalendar
									label={protocolOrReportLabel}
									labelClassName={stableFieldLabelClassName}
									value={addInspectionForm.dataProtokolu}
									previewTextOnly={isPreviewMode}
									onChange={(next) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataProtokolu: next,
										}))
									}
								/>

								{isControlType ? (
									<DateInputWithCalendar
										label={protocolDeliveryLabel}
										labelClassName={stableFieldLabelClassName}
										value={addInspectionForm.dataDoreczeniaProtokolu}
										previewTextOnly={isPreviewMode}
										onChange={(next) =>
											setAddInspectionForm((prev) => ({
												...prev,
												dataDoreczeniaProtokolu: next,
											}))
										}
									/>
								) : null}

								{isSupervisoryVisitType ? (
									<DateInputWithCalendar
										label={reportAcceptanceLabel}
										labelClassName={stableFieldLabelClassName}
										value={addInspectionForm.dataAkceptacjiSprawozdania}
										previewTextOnly={isPreviewMode}
										onChange={(next) =>
											setAddInspectionForm((prev) => ({
												...prev,
												dataAkceptacjiSprawozdania: next,
											}))
										}
									/>
								) : null}

								{isSupervisoryVisitType
									? renderNoLetterField({
											label: visitLetterDeliveryLabel,
											value: addInspectionForm.dataDoreczeniaPisma,
											isNoLetter: addInspectionForm.brakDataDoreczeniaPisma,
											onChangeValue: (nextValue) =>
												setAddInspectionForm((prev) => ({
													...prev,
													dataDoreczeniaPisma: nextValue,
													brakDataDoreczeniaPisma: nextValue
														? false
														: prev.brakDataDoreczeniaPisma,
												})),
											onChangeNoLetter: (nextNoLetter) =>
												setAddInspectionForm((prev) => ({
													...prev,
													brakDataDoreczeniaPisma: nextNoLetter,
													dataDoreczeniaPisma: nextNoLetter
														? ""
														: prev.dataDoreczeniaPisma,
												})),
									  })
									: null}

								{renderNoLetterField({
									label: objectionsLetterLabel,
									value: addInspectionForm.dataPismaZastrzezenia,
									isNoLetter: addInspectionForm.brakDataPismaZastrzezenia,
									onChangeValue: (nextValue) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataPismaZastrzezenia: nextValue,
											brakDataPismaZastrzezenia: nextValue ? false : prev.brakDataPismaZastrzezenia,
										})),
									onChangeNoLetter: (nextNoLetter) =>
										setAddInspectionForm((prev) => ({
											...prev,
											brakDataPismaZastrzezenia: nextNoLetter,
											dataPismaZastrzezenia: nextNoLetter ? "" : prev.dataPismaZastrzezenia,
										})),
								})}

								{renderNoLetterField({
									label: objectionsLetterSentLabel,
									value: addInspectionForm.dataWyslaniaPismaZZastrzezeniami,
									isNoLetter: addInspectionForm.brakDataWyslaniaPismaZZastrzezeniami,
									onChangeValue: (nextValue) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataWyslaniaPismaZZastrzezeniami: nextValue,
											brakDataWyslaniaPismaZZastrzezeniami: nextValue
												? false
												: prev.brakDataWyslaniaPismaZZastrzezeniami,
										})),
									onChangeNoLetter: (nextNoLetter) =>
										setAddInspectionForm((prev) => ({
											...prev,
											brakDataWyslaniaPismaZZastrzezeniami: nextNoLetter,
											dataWyslaniaPismaZZastrzezeniami: nextNoLetter
												? ""
												: prev.dataWyslaniaPismaZZastrzezeniami,
										})),
								})}

								{renderNoLetterField({
									label: objectionsReceivedLabel,
									value: addInspectionForm.dataWplywuPisma,
									isNoLetter: addInspectionForm.brakDataWplywuPisma,
									onChangeValue: (nextValue) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataWplywuPisma: nextValue,
											brakDataWplywuPisma: nextValue ? false : prev.brakDataWplywuPisma,
										})),
									onChangeNoLetter: (nextNoLetter) =>
										setAddInspectionForm((prev) => ({
											...prev,
											brakDataWplywuPisma: nextNoLetter,
											dataWplywuPisma: nextNoLetter ? "" : prev.dataWplywuPisma,
										})),
								})}

								{isControlType
									? renderNoLetterField({
											label: controlResponseLetterSentLabel,
											value: addInspectionForm.dataWyslaniaPismaZOdpowiedzia,
											isNoLetter: addInspectionForm.brakDataWyslaniaPismaZOdpowiedzia,
											onChangeValue: (nextValue) =>
												setAddInspectionForm((prev) => ({
													...prev,
													dataWyslaniaPismaZOdpowiedzia: nextValue,
													brakDataWyslaniaPismaZOdpowiedzia: nextValue
														? false
														: prev.brakDataWyslaniaPismaZOdpowiedzia,
												})),
											onChangeNoLetter: (nextNoLetter) =>
												setAddInspectionForm((prev) => ({
													...prev,
													brakDataWyslaniaPismaZOdpowiedzia: nextNoLetter,
													dataWyslaniaPismaZOdpowiedzia: nextNoLetter
														? ""
														: prev.dataWyslaniaPismaZOdpowiedzia,
												})),
									  })
									: null}

								{isControlType
									? renderNoLetterField({
											label: controlResponseLetterLabel,
											value: addInspectionForm.dataPismaZOdpowiedzia,
											isNoLetter: addInspectionForm.brakDataPismaZOdpowiedzia,
											onChangeValue: (nextValue) =>
												setAddInspectionForm((prev) => ({
													...prev,
													dataPismaZOdpowiedzia: nextValue,
													brakDataPismaZOdpowiedzia: nextValue
														? false
														: prev.brakDataPismaZOdpowiedzia,
												})),
											onChangeNoLetter: (nextNoLetter) =>
												setAddInspectionForm((prev) => ({
													...prev,
													brakDataPismaZOdpowiedzia: nextNoLetter,
													dataPismaZOdpowiedzia: nextNoLetter
														? ""
														: prev.dataPismaZOdpowiedzia,
												})),
									  })
									: null}

								<div>
									{isPreviewMode ? (
										<div>
											<label className={`mb-1 block text-slate-600 text-sm ${stableFieldLabelClassName}`}>
												Data akceptacji noty (lista)
											</label>
											<div className="text-slate-700 text-sm">
												{isDataAkceptacjiNotyBrak ? (
													<span className="font-semibold text-slate-900">Brak</span>
												) : parsedAcceptanceDates.length > 0 ? (
													<div className="space-y-0.5">
														{parsedAcceptanceDates.map((value, index) => (
															<div key={`acceptance-date-${value}-${index}`} className="font-semibold text-slate-900">{value}</div>
														))}
													</div>
												) : (
													<span className="font-semibold text-slate-900">-</span>
												)}
											</div>
										</div>
									) : (
										<DateListEditor
											title="Data akceptacji noty (lista)"
											addButtonLabel="Dodaj datę"
											noDatesLabel="Brak dat akceptacji noty"
											noDatesMessage="Oznaczono brak dat akceptacji noty."
											values={dataAkceptacjiNotyList}
											setValues={setDataAkceptacjiNotyList}
											isNoDates={isDataAkceptacjiNotyBrak}
											setIsNoDates={setIsDataAkceptacjiNotyBrak}
											itemKeyPrefix="akceptacja-noty"
										/>
									)}
								</div>

								{isPreviewMode ? (
									<div>
										<label className={`mb-1 block text-slate-600 text-sm ${stableFieldLabelClassName}`}>
											Data zaleceń
										</label>
										<div className="text-slate-700 text-sm">
											{parsedRecommendationDates.length > 0 ? (
												<div className="space-y-0.5">
													{parsedRecommendationDates.map((value, index) => (
														<div key={`recommendation-date-${value}-${index}`} className="font-semibold text-slate-900">{value}</div>
													))}
												</div>
											) : (
												<span className="font-semibold text-slate-900">-</span>
											)}
										</div>
									</div>
								) : null}

							</div>
							)}
						</section>
						</div>

					</fieldset>
			</div>
		</RegistryFormScaffold>
	);
}
