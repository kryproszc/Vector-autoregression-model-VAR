import { CalendarDays, Check, ChevronDown, ChevronsUpDown } from "lucide-react";
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

type InspectionsFormModalProps = {
	isOpen: boolean;
	editingInspectionId: string | null;
	editingInspectionCode?: string | null;
	showRequiredFieldErrors?: boolean;
	addInspectionForm: AddInspectionForm;
	setAddInspectionForm: Dispatch<SetStateAction<AddInspectionForm>>;
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
	errorMessage = null,
}: {
	label: string;
	value: string;
	onChange: (next: string) => void;
	disabled?: boolean;
	labelClassName?: string;
	invalid?: boolean;
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
	const [tempDate, setTempDate] = useState<Date | null>(() =>
		parseIsoDate(value) ?? null,
	);
	const popupWidth = calendarView === "day" ? 288 : 336;
	const popupHeight = calendarView === "day" ? 420 : 372;

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
		setTempDate(today);
		onChange(toIsoDateValue(today));
		setCalendarView("day");
		setIsCalendarOpen(false);
	};

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
										minDate={MIN_CALENDAR_DATE}
										maxDate={MAX_CALENDAR_DATE}
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
	options: Array<{ id: number; label: string }>;
	values: number[];
	onChange: (next: number[]) => void;
	disabled?: boolean;
	selectedSummary?: string;
	errorMessage?: string | null;
	highlightedUserId?: number | null;
}) {
	const [isOpen, setIsOpen] = useState(false);
	const [searchQuery, setSearchQuery] = useState("");
	const triggerRef = useRef<HTMLButtonElement | null>(null);
	const popupRef = useRef<HTMLDivElement | null>(null);
	const [popupPosition, setPopupPosition] = useState<{
		top: number;
		left: number;
		width: number;
	} | null>(null);

	const selectedCount = values.length;
	const normalizedSearchQuery = searchQuery.trim().toLowerCase();
	const visibleOptions = normalizedSearchQuery
		? options.filter((option) =>
				option.label.toLowerCase().includes(normalizedSearchQuery),
			)
		: options;

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
		const spaceBelow = availableBottom - rect.bottom;
		const shouldOpenUp =
			spaceBelow < popupHeight + 8 && rect.top > popupHeight + 8;

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
										onClick={() => onChange([])}
										className="font-semibold text-slate-600 text-xs hover:text-slate-900"
									>
										Wyczyść
									</button>
									<button
										type="button"
										onClick={() => onChange(options.map((option) => option.id))}
										className="font-semibold text-slate-600 text-xs hover:text-slate-900"
									>
										Zaznacz wszystkie
									</button>
								</div>

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

			{selectedSummary ? (
				<span className="mt-2 block text-slate-600 text-xs">Wybrano: {selectedSummary}</span>
			) : null}
			{errorMessage ? (
				<span className="mt-1 block text-rose-700 text-xs">{errorMessage}</span>
			) : null}
		</label>
	);
}

export function InspectionsFormModal({
	isOpen,
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
	const normalizedLeaderSearchQuery = leaderSearchQuery.trim().toLowerCase();
	const filteredLeaderUsers = normalizedLeaderSearchQuery
		? availableLeaderUsers.filter((user) =>
				getUserDisplayName(user)
					.toLowerCase()
					.includes(normalizedLeaderSearchQuery),
			)
		: availableLeaderUsers;
	const normalizedInspectionType = addInspectionForm.typInspekcji
		.trim()
		.toLowerCase();
	const isControlType = normalizedInspectionType.includes("kontrol");
	const isSupervisoryVisitType = normalizedInspectionType.includes("wizyta");
	const isControlOnlyFieldDisabled = !isControlType;
	const isVisitOnlyFieldDisabled = !isSupervisoryVisitType;
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
	const isRequiredInspectionTypeMissing =
		showRequiredFieldErrors && !addInspectionForm.typInspekcji.trim();
	const isRequiredEntityNameMissing =
		showRequiredFieldErrors && !addInspectionForm.nazwaPodmiotu.trim();
	const isRequiredStartDateMissing =
		showRequiredFieldErrors && !addInspectionForm.poczatekInspekcji;
	const isRequiredEndDateMissing =
		showRequiredFieldErrors && !addInspectionForm.koniecInspekcji;
	const isRequiredStatusMissing = showRequiredFieldErrors && !addInspectionForm.status.trim();

	return (
		<RegistryFormScaffold
			isOpen={isOpen}
			title={editingInspectionId ? "Edytuj inspekcję" : "Dodaj inspekcję"}
			subtitle={
				editingInspectionCode ? `Id inspekcji: ${editingInspectionCode}` : undefined
			}
			onClose={onClose}
			onSubmit={onSubmit}
			closeOnBackdropClick={false}
			maxWidthClassName="max-w-[1900px]"
			isContentReadOnly={isReadOnly}
			cancelLabel={editingInspectionId ? "Anuluj" : undefined}
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
			isSubmitDisabled={isSubmittingInspection || isReadOnly || isSaveDisabledDueToLock}
			submitLabel={
				isSubmittingInspection
					? "Zapisywanie..."
					: isReadOnly
						? "Tylko podgląd"
						: editingInspectionId
							? "Zapisz"
							: "Dodaj"
			}
		>
			<div className="space-y-4">
					<fieldset disabled={isReadOnly} className="border-0 p-0">
						<div className="grid gap-4 xl:grid-cols-12 xl:items-start">
						<section className="rounded-xl border border-slate-200 p-3 xl:col-span-5">
							<h4 className="mb-3 font-semibold text-slate-700 text-sm uppercase tracking-wide">
								Dane podstawowe
							</h4>

							<div className="grid gap-3 xl:grid-cols-2">
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

								<SingleSelectPortalField
									label="Nazwa podmiotu *"
									value={addInspectionForm.nazwaPodmiotu}
									options={entityNameOptions}
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
									<span className="mt-1 block text-slate-500 text-xs">
										Możesz wybrać wiele pozycji.
									</span>
								</label>

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

								<DateInputWithCalendar
									label="Początek inspekcji *"
									value={addInspectionForm.poczatekInspekcji}
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
									{isLeaderSelectionLocked ? (
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

								<div className="sm:col-span-2">
									<MultiSelectPeoplePortalField
										label="Skład zespołu"
										placeholder="Wybierz członków zespołu"
										options={activeUsers.map((user) => ({
											id: user.id,
											label: getUserDisplayName(user),
										}))}
										values={selectedTeamMemberIds}
										onChange={setSelectedTeamMemberIds}
										selectedSummary={selectedTeamMembers.join(", ")}
										errorMessage={teamMemberScopeError}
										highlightedUserId={outOfScopeTeamMemberUserId}
									/>
								</div>

							<label className="text-slate-700 text-sm sm:col-span-2">
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
							</div>

						</section>

						<section className="rounded-xl border border-slate-200 p-3 xl:col-span-7">
							<h4 className="mb-3 font-semibold text-slate-700 text-sm uppercase tracking-wide">
								Terminy
							</h4>

							<div className="grid gap-3 xl:grid-cols-2">
								<DateInputWithCalendar
									label={protocolOrReportLabel}
									labelClassName={stableFieldLabelClassName}
									value={addInspectionForm.dataProtokolu}
									onChange={(next) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataProtokolu: next,
										}))
									}
								/>

								<DateInputWithCalendar
									label={protocolDeliveryLabel}
									labelClassName={stableFieldLabelClassName}
									disabled={isControlOnlyFieldDisabled}
									value={addInspectionForm.dataDoreczeniaProtokolu}
									onChange={(next) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataDoreczeniaProtokolu: next,
										}))
									}
								/>

								<DateInputWithCalendar
									label={reportAcceptanceLabel}
									labelClassName={stableFieldLabelClassName}
									disabled={isVisitOnlyFieldDisabled}
									value={addInspectionForm.dataAkceptacjiSprawozdania}
									onChange={(next) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataAkceptacjiSprawozdania: next,
										}))
									}
								/>

								<NoLetterDateField
									label={visitLetterDeliveryLabel}
									labelClassName={stableFieldLabelClassName}
									value={addInspectionForm.dataDoreczeniaPisma}
									isNoLetter={addInspectionForm.brakDataDoreczeniaPisma}
									disabled={isVisitOnlyFieldDisabled}
									onChangeValue={(nextValue) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataDoreczeniaPisma: nextValue,
											brakDataDoreczeniaPisma: nextValue ? false : prev.brakDataDoreczeniaPisma,
										}))
									}
									onChangeNoLetter={(nextNoLetter) =>
										setAddInspectionForm((prev) => ({
											...prev,
											brakDataDoreczeniaPisma: nextNoLetter,
											dataDoreczeniaPisma: nextNoLetter ? "" : prev.dataDoreczeniaPisma,
										}))
									}
								/>

								<NoLetterDateField
									label={objectionsLetterLabel}
									labelClassName={stableFieldLabelClassName}
									value={addInspectionForm.dataPismaZastrzezenia}
									isNoLetter={addInspectionForm.brakDataPismaZastrzezenia}
									onChangeValue={(nextValue) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataPismaZastrzezenia: nextValue,
											brakDataPismaZastrzezenia: nextValue ? false : prev.brakDataPismaZastrzezenia,
										}))
									}
									onChangeNoLetter={(nextNoLetter) =>
										setAddInspectionForm((prev) => ({
											...prev,
											brakDataPismaZastrzezenia: nextNoLetter,
											dataPismaZastrzezenia: nextNoLetter ? "" : prev.dataPismaZastrzezenia,
										}))
									}
								/>

								<DateInputWithCalendar
									label={objectionsLetterSentLabel}
									labelClassName={stableFieldLabelClassName}
									value={addInspectionForm.dataWyslaniaPismaZZastrzezeniami}
									onChange={(next) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataWyslaniaPismaZZastrzezeniami: next,
										}))
									}
								/>

								<NoLetterDateField
									label={objectionsReceivedLabel}
									labelClassName={stableFieldLabelClassName}
									value={addInspectionForm.dataWplywuPisma}
									isNoLetter={addInspectionForm.brakDataWplywuPisma}
									onChangeValue={(nextValue) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataWplywuPisma: nextValue,
											brakDataWplywuPisma: nextValue ? false : prev.brakDataWplywuPisma,
										}))
									}
									onChangeNoLetter={(nextNoLetter) =>
										setAddInspectionForm((prev) => ({
											...prev,
											brakDataWplywuPisma: nextNoLetter,
											dataWplywuPisma: nextNoLetter ? "" : prev.dataWplywuPisma,
										}))
									}
								/>

								<DateInputWithCalendar
									label={controlResponseLetterSentLabel}
									labelClassName={stableFieldLabelClassName}
									disabled={isControlOnlyFieldDisabled}
									value={addInspectionForm.dataWyslaniaPismaZOdpowiedzia}
									onChange={(next) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataWyslaniaPismaZOdpowiedzia: next,
										}))
									}
								/>

								<NoLetterDateField
									label={controlResponseLetterLabel}
									labelClassName={stableFieldLabelClassName}
									value={addInspectionForm.dataPismaZOdpowiedzia}
									isNoLetter={addInspectionForm.brakDataPismaZOdpowiedzia}
									disabled={isControlOnlyFieldDisabled}
									onChangeValue={(nextValue) =>
										setAddInspectionForm((prev) => ({
											...prev,
											dataPismaZOdpowiedzia: nextValue,
											brakDataPismaZOdpowiedzia: nextValue ? false : prev.brakDataPismaZOdpowiedzia,
										}))
									}
									onChangeNoLetter={(nextNoLetter) =>
										setAddInspectionForm((prev) => ({
											...prev,
											brakDataPismaZOdpowiedzia: nextNoLetter,
											dataPismaZOdpowiedzia: nextNoLetter ? "" : prev.dataPismaZOdpowiedzia,
										}))
									}
								/>

								<div className="xl:col-start-2">
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
								</div>

							</div>
						</section>
						</div>

					</fieldset>
			</div>
		</RegistryFormScaffold>
	);
}
