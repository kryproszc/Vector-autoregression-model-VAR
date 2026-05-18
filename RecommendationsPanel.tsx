"use client";

import {
	CalendarDays,
	Pencil,
	Plus,
	Trash2,
} from "lucide-react";
import type { SetStateAction } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { DateCalendar } from "@mui/x-date-pickers/DateCalendar";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { AdapterDateFns } from "@mui/x-date-pickers/AdapterDateFns";
import { pl } from "date-fns/locale";
import type { AuthRole } from "@/app/_components/home-tabs/types";

import { fetchDictionaryEntries } from "@/features/dictionaries/api";
import type { DictionaryEntry } from "@/features/dictionaries/types";
import {
	type RawInspectionRow,
	normalizeInspectionRow,
} from "@/features/inspections/components/inspections-panel.utils";
import { fetchObligatingDecisions } from "@/features/obligating-decisions/api";
import { RecommendationsSuccessModal } from "@/features/recommendations/components/RecommendationsSuccessModal";
import {
	createRecommendation,
	deleteRecommendation,
	fetchRecommendations,
	type RecommendationLockConflict,
	updateRecommendation,
} from "@/features/recommendations/api";
import { fetchSanctionRequests } from "@/features/sanction-requests/api";
import type {
	RecommendationRead,
	RecommendationWrite,
} from "@/features/recommendations/types";
import { DateListEditor } from "@/shared/components/forms/DateListEditor";
import { RegistryFormScaffold } from "@/shared/components/forms/RegistryFormScaffold";
import { SingleSelectPortalField } from "@/shared/components/forms/SingleSelectPortalField";
import { toDateList } from "@/shared/utils/date";
import { DeleteSuccessModal } from "@/shared/components/DeleteSuccessModal";
import { ExportConfigModal } from "@/shared/components/export/ExportConfigModal";
import { TableAdvancedFilterModal } from "@/shared/components/table/TableAdvancedFilterModal";
import { TableColumnPickerModal } from "@/shared/components/table/TableColumnPickerModal";
import { TableHeaderWithFilters } from "@/shared/components/table/TableHeaderWithFilters";
import { TablePanelToolbar } from "@/shared/components/table/TablePanelToolbar";
import { TablePagination } from "@/shared/components/table/TablePagination";
import { TableSurface } from "@/shared/components/table/TableSurface";
import {
	addWorksheetWithStyles,
	createStyledExportWorkbook,
	saveWorkbookAsXlsx,
} from "@/shared/utils/excel-export";
import { getFloatingPanelAnchor } from "@/shared/utils/floating-panel";
import { useTableState } from "@/shared/hooks/useTableState";
import { useInactivityTimeout } from "@/shared/hooks/useInactivityTimeout";
import { useRecordLock } from "@/shared/hooks/useRecordLock";

const INACTIVITY_TIMEOUT_MS = 60_000; // 1 minuta (do testów)
const INACTIVITY_WARNING_MS = 30_000; // 30 sekund ostrzeżenia
const TABLE_PAGE_SIZE_OPTIONS = [20, 30, 50, 70, 100] as const;
const NO_DATES_MARKER = "Brak";

type RecommendationsPanelProps = {
	operatorLogin: string;
	authRole: AuthRole;
	isObserver?: boolean;
};

const RECOMMENDATIONS_CHANGED_EVENT = "recommendations:changed";
const INSPECTIONS_CHANGED_EVENT = "inspections:changed";
const DASHBOARD_OPEN_RECOMMENDATION_EVENT = "dashboard:open-recommendation";
const DASHBOARD_OPEN_RECOMMENDATION_CODE_KEY =
	"triangle.dashboard.openRecommendationCode";

type RecommendationColumnKey =
	| "lp"
	| "kodZalecenia"
	| "pozycja"
	| "inspectionId"
	| "nazwaPodmiotu"
	| "terminWykonaniaZalecen"
	| "status"
	| "komentarz"
	| "dataZalecenList"
	| "dataAkceptacjiNotyWeryfikacjiList";

type RecommendationColumn = {
	key: RecommendationColumnKey;
	label: string;
};

const RECOMMENDATION_COLUMNS: RecommendationColumn[] = [
	{ key: "lp", label: "Lp." },
	{ key: "kodZalecenia", label: "Id zalecenia" },
	{ key: "inspectionId", label: "Id inspekcji" },
	{ key: "nazwaPodmiotu", label: "Nazwa podmiotu" },
	{ key: "terminWykonaniaZalecen", label: "Data zaleceń" },
	{ key: "dataZalecenList", label: "Termin wykonania zaleceń" },
	{ key: "pozycja", label: "Liczba zaleceń" },
	{
		key: "dataAkceptacjiNotyWeryfikacjiList",
		label: "Data akceptacji noty z weryfikacji wykonania zaleceń",
	},
	{ key: "status", label: "Status" },
	{ key: "komentarz", label: "Komentarz" },
];

const RECOMMENDATION_COLUMN_TOOLTIPS: Partial<
	Record<RecommendationColumnKey, string>
> = {
	kodZalecenia: "Unikalne id zalecenia",
	inspectionId: "Unikalne id inspekcji",
};

const ALL_RECOMMENDATION_COLUMN_KEYS: RecommendationColumnKey[] =
	RECOMMENDATION_COLUMNS.map((column) => column.key);

type RecommendationFormState = {
	inspectionId: string;
	isInspectionMissing: boolean;
	pozycja: string;
	nazwaPodmiotu: string;
	terminWykonaniaZalecen: string;
	status: string;
	komentarz: string;
	dataZalecenList: string[];
	dataAkceptacjiList: string[];
	isDataZalecenBrak: boolean;
	isDataAkceptacjiBrak: boolean;
};

type InspectionOption = {
	id: number;
	lp: number;
	inspectionCode: string;
	nazwaPodmiotu: string;
};

const INSPECTIONS_API_URL = "/api/structure/inspections";
const AVAILABLE_INSPECTIONS_API_URL = "/api/recommendations/available-inspections";

type InspectionExportColumnKey =
	| "kodInspekcji"
	| "nazwaPodmiotu"
	| "typInspekcji"
	| "zakresInspekcji"
	| "aspektKonsumencki"
	| "poczatekInspekcji"
	| "koniecInspekcji"
	| "osobaKierujaca"
	| "skladZespolu"
	| "rynek"
	| "rodzajPodmiotu"
	| "dataProtokolu"
	| "dataDoreczeniaProtokolu"
	| "dataAkceptacjiSprawozdania"
	| "dataDoreczeniaPisma"
	| "dataPismaZastrzezenia"
	| "dataWyslaniaPismaZZastrzezeniami"
	| "dataWplywuPisma"
	| "dataPismaZOdpowiedzia"
	| "dataWyslaniaPismaZOdpowiedzia"
	| "dataAkceptacjiNoty"
	| "dataZalecen"
	| "status"
	| "komentarz";

type SanctionExportColumnKey =
	| "lp"
	| "requestId"
	| "inspectionLp"
	| "nazwaPodmiotuObjetegoInspekcja"
	| "nazwaPodmiotuObjetegoSankcjaList"
	| "dataWniosku"
	| "wniosekDo"
	| "sankcjaList"
	| "podstawaPrawnaSankcjiList"
	| "naruszeniaSkutkujaceSankcjaList"
	| "czyMamyInformacjeOWszczeciuPostepowania"
	| "rozstrzygniecie"
	| "komentarz";

type DecisionExportColumnKey =
	| "lp"
	| "kodDecyzji"
	| "kodZalecenia"
	| "inspectionLp"
	| "nazwaPodmiotu"
	| "liczbaZalecen"
	| "dataWszczeciaPostepowaniaIInstancji"
	| "osobyProwadzaceIInstancjeList"
	| "dataDecyzjiIInstancji"
	| "dataDoreczeniaDecyzjiIInstancji"
	| "rozstrzygniecieI"
	| "dataWnioskuPonowneRozpatrzenie"
	| "dataWplywuWnioskuPonowneRozpatrzenie"
	| "osobyProwadzaceIIInstancjeList"
	| "dataDecyzjiIIInstancji"
	| "dataDoreczeniaDecyzjiIIInstancji"
	| "rozstrzygniecieII"
	| "komentarz";

type ExportColumnDefinition<T extends string> = {
	key: T;
	label: string;
};

const INSPECTION_EXPORT_COLUMNS: ExportColumnDefinition<InspectionExportColumnKey>[] = [
	{ key: "kodInspekcji", label: "Id inspekcji" },
	{ key: "nazwaPodmiotu", label: "Nazwa podmiotu" },
	{ key: "typInspekcji", label: "Typ inspekcji" },
	{ key: "zakresInspekcji", label: "Zakres inspekcji" },
	{ key: "aspektKonsumencki", label: "Aspekt konsumencki" },
	{ key: "poczatekInspekcji", label: "Początek" },
	{ key: "koniecInspekcji", label: "Koniec" },
	{ key: "osobaKierujaca", label: "Osoba kierująca" },
	{ key: "skladZespolu", label: "Skład zespołu" },
	{ key: "rynek", label: "Rynek" },
	{ key: "rodzajPodmiotu", label: "Rodzaj podmiotu" },
	{ key: "dataProtokolu", label: "Data protokołu / sprawozdania" },
	{ key: "dataDoreczeniaProtokolu", label: "Data doręczenia protokołu" },
	{
		key: "dataAkceptacjiSprawozdania",
		label: "Data akceptacji sprawozdania",
	},
	{ key: "dataDoreczeniaPisma", label: "Data doręczenia pisma" },
	{ key: "dataPismaZastrzezenia", label: "Data pisma zastrzeżenia" },
	{
		key: "dataWyslaniaPismaZZastrzezeniami",
		label: "Data wysłania pisma z zastrzeżeniami",
	},
	{ key: "dataWplywuPisma", label: "Data wpływu pisma" },
	{ key: "dataPismaZOdpowiedzia", label: "Data pisma z odpowiedzią" },
	{
		key: "dataWyslaniaPismaZOdpowiedzia",
		label: "Data wysłania pisma z odpowiedzią",
	},
	{ key: "dataAkceptacjiNoty", label: "Data akceptacji noty" },
	{ key: "dataZalecen", label: "Data zaleceń" },
	{ key: "status", label: "Status" },
	{ key: "komentarz", label: "Komentarz" },
];

const SANCTION_EXPORT_COLUMNS: ExportColumnDefinition<SanctionExportColumnKey>[] = [
	{ key: "lp", label: "Lp. wniosku" },
	{ key: "requestId", label: "Id wniosku" },
	{ key: "inspectionLp", label: "Id inspekcji" },
	{
		key: "nazwaPodmiotuObjetegoInspekcja",
		label: "Nazwa podmiotu objętego inspekcją",
	},
	{
		key: "nazwaPodmiotuObjetegoSankcjaList",
		label: "Nazwa podmiotu objętego sankcją",
	},
	{ key: "dataWniosku", label: "Data wniosku" },
	{ key: "wniosekDo", label: "Wniosek do" },
	{ key: "sankcjaList", label: "Sankcja" },
	{ key: "podstawaPrawnaSankcjiList", label: "Podstawa prawna sankcji" },
	{
		key: "naruszeniaSkutkujaceSankcjaList",
		label: "Naruszenia skutkujące sankcją",
	},
	{
		key: "czyMamyInformacjeOWszczeciuPostepowania",
		label: "Informacja o wszczęciu postępowania",
	},
	{ key: "rozstrzygniecie", label: "Rozstrzygnięcie" },
	{ key: "komentarz", label: "Komentarz" },
];

const DECISION_EXPORT_COLUMNS: ExportColumnDefinition<DecisionExportColumnKey>[] = [
	{ key: "lp", label: "Lp. decyzji" },
	{ key: "kodDecyzji", label: "Id decyzji" },
	{ key: "kodZalecenia", label: "Id zalecenia" },
	{ key: "inspectionLp", label: "Id inspekcji" },
	{ key: "nazwaPodmiotu", label: "Nazwa podmiotu" },
	{ key: "liczbaZalecen", label: "Liczba zaleceń" },
	{
		key: "dataWszczeciaPostepowaniaIInstancji",
		label: "Data wszczęcia postępowania I instancji",
	},
	{ key: "osobyProwadzaceIInstancjeList", label: "Osoby prowadzące I instancję" },
	{ key: "dataDecyzjiIInstancji", label: "Data decyzji I instancji" },
	{
		key: "dataDoreczeniaDecyzjiIInstancji",
		label: "Data doręczenia decyzji I instancji",
	},
	{ key: "rozstrzygniecieI", label: "Rozstrzygnięcie decyzji I instancji" },
	{
		key: "dataWnioskuPonowneRozpatrzenie",
		label: "Data wniosku o ponowne rozpatrzenie",
	},
	{
		key: "dataWplywuWnioskuPonowneRozpatrzenie",
		label: "Data wpływu wniosku o ponowne rozpatrzenie",
	},
	{ key: "osobyProwadzaceIIInstancjeList", label: "Osoby prowadzące II instancję" },
	{ key: "dataDecyzjiIIInstancji", label: "Data decyzji II instancji" },
	{
		key: "dataDoreczeniaDecyzjiIIInstancji",
		label: "Data doręczenia decyzji II instancji",
	},
	{ key: "rozstrzygniecieII", label: "Rozstrzygnięcie decyzji II instancji" },
	{ key: "komentarz", label: "Komentarz" },
];

const EMPTY_FORM: RecommendationFormState = {
	inspectionId: "",
	isInspectionMissing: false,
	pozycja: "",
	nazwaPodmiotu: "",
	terminWykonaniaZalecen: "",
	status: "",
	komentarz: "",
	dataZalecenList: [],
	dataAkceptacjiList: [],
	isDataZalecenBrak: false,
	isDataAkceptacjiBrak: false,
};

function resolveSetStateAction<T>(
	nextValue: SetStateAction<T>,
	prevValue: T,
): T {
	if (typeof nextValue === "function") {
		return (nextValue as (prev: T) => T)(prevValue);
	}

	return nextValue;
}

function mapDictionaryEntriesToOptions(entries: DictionaryEntry[]) {
	const mappedOptions = entries
		.filter((entry) => entry.aktywny)
		.sort((left, right) => {
			const leftOrder = left.kolejnosc ?? Number.MAX_SAFE_INTEGER;
			const rightOrder = right.kolejnosc ?? Number.MAX_SAFE_INTEGER;
			if (leftOrder !== rightOrder) {
				return leftOrder - rightOrder;
			}

			return left.nazwaPozycji.localeCompare(right.nazwaPozycji, "pl", {
				sensitivity: "base",
			});
		})
		.map((entry) => entry.nazwaPozycji.trim())
		.filter(Boolean);

	return Array.from(new Set(mappedOptions));
}

function formatDateListDisplay(values: string[], isNoDatesSelected: boolean) {
	if (isNoDatesSelected) {
		return NO_DATES_MARKER;
	}

	const normalizedDates = toDateList(values);
	if (normalizedDates.length === 0) {
		return "-";
	}

	return normalizedDates.join(", ");
}

function formToPayload(
	form: RecommendationFormState,
): RecommendationWrite | null {
	const inspectionId = Number(form.inspectionId);
	const pozycja = Number(form.pozycja);

	if (!Number.isFinite(pozycja) || pozycja <= 0) {
		return null;
	}

	if (
		!form.isInspectionMissing &&
		(!Number.isFinite(inspectionId) || inspectionId <= 0)
	) {
		return null;
	}

	const normalizedListDates = form.isDataZalecenBrak
		? []
		: toDateList(form.dataZalecenList);
	const singleDate = form.terminWykonaniaZalecen.trim() || null;

	return {
		inspectionId: form.isInspectionMissing ? null : inspectionId,
		pozycja,
		nazwaPodmiotu: form.nazwaPodmiotu.trim(),
		dataZalecen: singleDate,
		terminyWykonaniaZalecenList: normalizedListDates,
		brakTerminowWykonaniaZalecen: form.isDataZalecenBrak,
		brakDatAkceptacjiNotyWeryfikacji: form.isDataAkceptacjiBrak,
		terminWykonaniaZalecen: singleDate,
		status: form.status.trim() || null,
		komentarz: form.komentarz.trim() || null,
		dataZalecenList: normalizedListDates,
		dataAkceptacjiNotyWeryfikacjiList: form.isDataAkceptacjiBrak
			? []
			: toDateList(form.dataAkceptacjiList),
	};
}

function recommendationToForm(
	item: RecommendationRead,
): RecommendationFormState {
	const hasInspectionLink =
		typeof item.inspectionId === "number" &&
		Number.isFinite(item.inspectionId) &&
		item.inspectionId > 0;
	const normalizedStatus = String(item.status ?? "").trim();
	const statusValue =
		normalizedStatus.toLowerCase() === "brak" ? "" : normalizedStatus;

	return {
		inspectionId: hasInspectionLink ? String(item.inspectionId) : "",
		isInspectionMissing: !hasInspectionLink,
		pozycja: String(item.pozycja),
		nazwaPodmiotu: item.nazwaPodmiotu,
		terminWykonaniaZalecen: item.dataZalecen ?? item.terminWykonaniaZalecen ?? "",
		status: statusValue,
		komentarz: item.komentarz ?? "",
		dataZalecenList: toDateList(
			item.terminyWykonaniaZalecenList.length > 0
				? item.terminyWykonaniaZalecenList
				: item.dataZalecenList,
		),
		dataAkceptacjiList: toDateList(item.dataAkceptacjiNotyWeryfikacjiList),
		isDataZalecenBrak: item.brakTerminowWykonaniaZalecen === true,
		isDataAkceptacjiBrak: item.brakDatAkceptacjiNotyWeryfikacji === true,
	};
}

function formatLockStartHourMinute(value: string | null | undefined) {
	if (!value) {
		return "--:--";
	}

	const date = new Date(value);
	if (Number.isNaN(date.getTime())) {
		return "--:--";
	}

	return new Intl.DateTimeFormat("pl-PL", {
		hour: "2-digit",
		minute: "2-digit",
		hour12: false,
	}).format(date);
}


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

function DateFieldWithClear({
	label,
	value,
	onChange,
	disabled = false,
}: {
	label: string;
	value: string;
	onChange: (next: string) => void;
	disabled?: boolean;
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
		const spaceBelow = window.innerHeight - anchorRect.bottom;
		const canOpenUp =
			spaceBelow < popupHeight + offset && anchorRect.top > popupHeight + offset;
		const top = canOpenUp
			? anchorRect.top - popupHeight - offset
			: anchorRect.bottom + offset;
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
			<span className="mb-1 block overflow-hidden text-ellipsis whitespace-nowrap">
				{label}
			</span>
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
					className="w-full cursor-pointer rounded-lg border border-slate-300 px-3 py-2 pr-10 text-sm outline-none transition-colors focus:border-blue-400 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-700"
				/>
				<button
					type="button"
					aria-label={`Otwórz kalendarz dla pola: ${label}`}
					disabled={disabled}
					onMouseDown={(event) => {
						event.preventDefault();
						event.stopPropagation();
					}}
					onClick={(event) => {
						event.preventDefault();
						event.stopPropagation();
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
		</label>
	);
}

export function RecommendationsPanel({
	operatorLogin,
	authRole,
	isObserver,
}: RecommendationsPanelProps) {
	const [items, setItems] = useState<RecommendationRead[]>([]);
	const [total, setTotal] = useState(0);
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [advancedFilterAnchor, setAdvancedFilterAnchor] = useState({
		top: 120,
		left: 120,
	});
	const [isExporting, setIsExporting] = useState(false);
	const [isExportConfigModalOpen, setIsExportConfigModalOpen] = useState(false);
	const [includeInspectionsInExport, setIncludeInspectionsInExport] =
		useState(false);
	const [includeSanctionsInExport, setIncludeSanctionsInExport] = useState(false);
	const [includeDecisionsInExport, setIncludeDecisionsInExport] = useState(false);
	const [activeExportColumnsTab, setActiveExportColumnsTab] = useState<
		"inspections" | "sanctions" | "decisions"
	>("inspections");
	const [selectedInspectionExportColumns, setSelectedInspectionExportColumns] =
		useState<InspectionExportColumnKey[]>(
			INSPECTION_EXPORT_COLUMNS.map((column) => column.key),
		);
	const [selectedSanctionExportColumns, setSelectedSanctionExportColumns] =
		useState<SanctionExportColumnKey[]>(
			SANCTION_EXPORT_COLUMNS.map((column) => column.key),
		);
	const [selectedDecisionExportColumns, setSelectedDecisionExportColumns] =
		useState<DecisionExportColumnKey[]>(
			DECISION_EXPORT_COLUMNS.map((column) => column.key),
		);

	const [selectedId, setSelectedId] = useState<number | null>(null);
	const [pendingDashboardRecommendationCode, setPendingDashboardRecommendationCode] =
		useState<string | null>(null);
	const [isFormOpen, setIsFormOpen] = useState(false);
	const [editingItem, setEditingItem] = useState<RecommendationRead | null>(
		null,
	);
	const [form, setForm] = useState<RecommendationFormState>(EMPTY_FORM);
	const [formError, setFormError] = useState<string | null>(null);
	const [showRequiredFieldErrors, setShowRequiredFieldErrors] = useState(false);
	const [versionConflictUpdatedAt, setVersionConflictUpdatedAt] = useState<string | null>(null);
	const [saveLockConflict, setSaveLockConflict] =
		useState<RecommendationLockConflict | null>(null);
	const [isSubmitting, setIsSubmitting] = useState(false);
	const [isSuccessModalOpen, setIsSuccessModalOpen] = useState(false);
	const [successEntityName, setSuccessEntityName] = useState("");
	const [successInspectionCode, setSuccessInspectionCode] = useState("");
	const [successMode, setSuccessMode] = useState<"create" | "edit">(
		"create",
	);
	const didNormalizeEditStatusRef = useRef(false);
	const [isDeleteConfirmModalOpen, setIsDeleteConfirmModalOpen] =
		useState(false);
	const [isDeletingItem, setIsDeletingItem] = useState(false);
	const [isDeleteSuccessModalOpen, setIsDeleteSuccessModalOpen] =
		useState(false);
	const [deleteSuccessEntityName, setDeleteSuccessEntityName] = useState("");
	const [tablePageSize, setTablePageSize] = useState<number>(30);
	const [inspectionOptions, setInspectionOptions] = useState<
		InspectionOption[]
	>([]);
	const [isInspectionOptionsLoading, setIsInspectionOptionsLoading] =
		useState(false);
	const [recommendationStatusOptions, setRecommendationStatusOptions] =
		useState<string[]>([]);
	const [entityNameOptions, setEntityNameOptions] = useState<string[]>([]);
	const canManageRecommendations = authRole !== "external_user" && !isObserver;
	const isDirector = authRole === "director";

	const selectedItem = useMemo(
		() => items.find((item) => item.id === selectedId) ?? null,
		[items, selectedId],
	);

	const selectedInspectionOption = useMemo(
		() =>
			inspectionOptions.find(
				(option) => String(option.id) === form.inspectionId,
			) ?? null,
		[form.inspectionId, inspectionOptions],
	);

	const inspectionSelectOptions = useMemo(
		() =>
			inspectionOptions.map((option) => ({
				value: String(option.id),
				label: `${option.inspectionCode}${
					option.nazwaPodmiotu ? ` - ${option.nazwaPodmiotu}` : ""
				}`,
			})),
		[inspectionOptions],
	);

	const isEditMode = Boolean(editingItem);
	const editRecordLock = useRecordLock({
		enabled: isFormOpen && isEditMode,
		module: "recommendations",
		recordId: editingItem?.id ?? null,
		operatorLogin,
		heartbeatIntervalMs: 20_000,
	});
	const shouldShowLockedByOtherUser = Boolean(saveLockConflict) || editRecordLock.isBlocked;
	const isReadOnlyDueToLock = isEditMode && shouldShowLockedByOtherUser;
	const isSaveDisabledDueToLock = isEditMode && (editRecordLock.isAcquireFailed || editRecordLock.isConnectionLost || editRecordLock.isExpired);
	const lockOwnerDisplayName =
		saveLockConflict?.ownerDisplayName || editRecordLock.owner?.displayName || "";
	const lockOwnerLogin =
		saveLockConflict?.ownerLogin || editRecordLock.owner?.login || "";
	const lockOwnerLabel =
		lockOwnerDisplayName || lockOwnerLogin
			? `${lockOwnerDisplayName || "Nieznany użytkownik"}${
					lockOwnerLogin ? ` (${lockOwnerLogin})` : ""
			  }`
			: "inny użytkownik";
	const lockAcquiredAt =
		saveLockConflict?.acquiredAt || editRecordLock.lockDetails?.acquiredAt || null;

	const closeModalRef = useRef<() => void>(() => {});
	const inactivityTimeout = useInactivityTimeout({
		enabled: isFormOpen,
		inactivityMs: INACTIVITY_TIMEOUT_MS,
		warningMs: INACTIVITY_WARNING_MS,
		onTimeout: () => closeModalRef.current(),
	});

	const inspectionCodeById = useMemo(
		() => new Map(inspectionOptions.map((option) => [option.id, option.inspectionCode])),
		[inspectionOptions],
	);

	const resolveInspectionCode = (payload: {
		inspectionKod?: unknown;
		kodInspekcji?: unknown;
		inspectionLp?: unknown;
		lp?: unknown;
		inspectionId?: unknown;
	}) => {
		const inspectionKod = String(payload.inspectionKod ?? "").trim();
		if (inspectionKod) {
			return inspectionKod;
		}

		const kodInspekcji = String(payload.kodInspekcji ?? "").trim();
		if (kodInspekcji) {
			return kodInspekcji;
		}

		const inspectionLp = String(payload.inspectionLp ?? "").trim();
		if (inspectionLp) {
			return inspectionLp;
		}

		const lp = String(payload.lp ?? "").trim();
		if (lp) {
			return lp;
		}

		const numericInspectionId = Number(payload.inspectionId);
		if (Number.isFinite(numericInspectionId) && numericInspectionId > 0) {
			const mappedCode = inspectionCodeById.get(numericInspectionId);
			if (mappedCode) {
				return mappedCode;
			}
			return String(numericInspectionId);
		}

		return "";
	};

	const getCellValue = (
		item: RecommendationRead,
		columnKey: RecommendationColumnKey,
	) => {
		if (columnKey === "terminWykonaniaZalecen") {
			return item.dataZalecen ?? item.terminWykonaniaZalecen ?? "";
		}

		if (columnKey === "inspectionId") {
			return resolveInspectionCode({
				inspectionKod: item.inspectionKod,
				kodInspekcji: item.kodInspekcji,
				inspectionLp: item.inspectionLp,
				inspectionId: item.inspectionId,
			});
		}

		if (columnKey === "kodZalecenia") {
			return String(item.kodZalecenia ?? "").trim();
		}

		if (columnKey === "dataZalecenList") {
			const source = item.terminyWykonaniaZalecenList.length > 0
				? item.terminyWykonaniaZalecenList
				: item.dataZalecenList;
			return formatDateListDisplay(
				source,
				item.brakTerminowWykonaniaZalecen === true,
			);
		}

		if (columnKey === "dataAkceptacjiNotyWeryfikacjiList") {
			return formatDateListDisplay(
				item.dataAkceptacjiNotyWeryfikacjiList,
				item.brakDatAkceptacjiNotyWeryfikacji === true,
			);
		}

		const raw = item[columnKey as keyof RecommendationRead];
		if (raw === null || raw === undefined) {
			return "";
		}

		return String(raw);
	};

	const setFormDataZalecenList = (nextValue: SetStateAction<string[]>) => {
		setForm((prev) => ({
			...prev,
			dataZalecenList: resolveSetStateAction(nextValue, prev.dataZalecenList),
		}));
	};

	const setFormDataAkceptacjiList = (nextValue: SetStateAction<string[]>) => {
		setForm((prev) => ({
			...prev,
			dataAkceptacjiList: resolveSetStateAction(
				nextValue,
				prev.dataAkceptacjiList,
			),
		}));
	};

	const setFormIsDataZalecenBrak = (nextValue: SetStateAction<boolean>) => {
		setForm((prev) => ({
			...prev,
			isDataZalecenBrak: resolveSetStateAction(
				nextValue,
				prev.isDataZalecenBrak,
			),
		}));
	};

	const setFormIsDataAkceptacjiBrak = (nextValue: SetStateAction<boolean>) => {
		setForm((prev) => ({
			...prev,
			isDataAkceptacjiBrak: resolveSetStateAction(
				nextValue,
				prev.isDataAkceptacjiBrak,
			),
		}));
	};

	const statusOptionsForForm = useMemo(() => {
		const normalizedCurrentStatus = form.status.trim();
		if (!normalizedCurrentStatus) {
			return recommendationStatusOptions;
		}

		return recommendationStatusOptions.includes(normalizedCurrentStatus)
			? recommendationStatusOptions
			: [normalizedCurrentStatus, ...recommendationStatusOptions];
	}, [form.status, recommendationStatusOptions]);

	useEffect(() => {
		if (!isFormOpen || !isEditMode) {
			didNormalizeEditStatusRef.current = false;
			return;
		}

		if (didNormalizeEditStatusRef.current) {
			return;
		}

		didNormalizeEditStatusRef.current = true;
		if (form.status.trim().toLowerCase() === "brak") {
			setForm((prev) => ({ ...prev, status: "" }));
		}
	}, [form.status, isEditMode, isFormOpen]);

	const {
		advancedFilterColumnKey,
		advancedFilterSearch,
		advancedFilters,
		canClearFilters,
		clearAdvancedFilterForSelectedColumn,
		clearFilters,
		columnFilters,
		draftHiddenColumns,
		draftVisibleColumns: draftVisibleRecommendationColumns,
		filteredAndSortedRows: filteredAndSortedItems,
		paginatedRows: paginatedRecommendationItems,
		currentPage,
		totalPages,
		pageSize,
		paginationItems,
		handlePageChange,
		handleApplyViewChanges,
		handleDraftColumnVisibilityChange,
		handleFilterChange,
		handleOpenViewModal,
		handleSortByColumn,
		isAdvancedFilterModalOpen,
		isColumnPickerOpen,
		selectedAdvancedFilterValues,
		selectAllVisibleAdvancedFilterValues,
		setAdvancedFilterColumnKey,
		setAdvancedFilterSearch,
		setDraftHiddenColumns,
		setIsAdvancedFilterModalOpen,
		setIsColumnPickerOpen,
		sortColumnKey,
		sortDirection,
		toggleAdvancedFilterValue,
		visibleAdvancedFilterValues,
		visibleColumns: visibleRecommendationColumns,
	} = useTableState<RecommendationRead, RecommendationColumnKey>({
		rows: items,
		allColumnKeys: ALL_RECOMMENDATION_COLUMN_KEYS,
		initialAdvancedFilterColumnKey: "nazwaPodmiotu",
		getCellValue,
		pageSize: tablePageSize,
		sortComparators: {
			lp: (left, right) => (Number(getCellValue(left, "lp")) || 0) - (Number(getCellValue(right, "lp")) || 0),
			pozycja: (left, right) =>
				(Number(getCellValue(left, "pozycja")) || 0) -
				(Number(getCellValue(right, "pozycja")) || 0),
		},
	});

	const handlePageSizeChange = (nextPageSize: number) => {
		if (
			!TABLE_PAGE_SIZE_OPTIONS.includes(
				nextPageSize as (typeof TABLE_PAGE_SIZE_OPTIONS)[number],
			)
		) {
			return;
		}

		setTablePageSize(nextPageSize);
	};

	useEffect(() => {
		if (typeof window === "undefined") {
			return;
		}

		const fromSession = window.sessionStorage.getItem(
			DASHBOARD_OPEN_RECOMMENDATION_CODE_KEY,
		);
		if (fromSession?.trim()) {
			setPendingDashboardRecommendationCode(fromSession.trim());
		}

		const handleOpenRecommendationFromDashboard = (event: Event) => {
			const customEvent = event as CustomEvent<{ recommendationCode?: unknown }>;
			const recommendationCode =
				typeof customEvent.detail?.recommendationCode === "string"
					? customEvent.detail.recommendationCode.trim()
					: "";
			if (!recommendationCode) {
				return;
			}

			window.sessionStorage.setItem(
				DASHBOARD_OPEN_RECOMMENDATION_CODE_KEY,
				recommendationCode,
			);
			setPendingDashboardRecommendationCode(recommendationCode);
		};

		window.addEventListener(
			DASHBOARD_OPEN_RECOMMENDATION_EVENT,
			handleOpenRecommendationFromDashboard,
		);

		return () => {
			window.removeEventListener(
				DASHBOARD_OPEN_RECOMMENDATION_EVENT,
				handleOpenRecommendationFromDashboard,
			);
		};
	}, []);

	useEffect(() => {
		if (!pendingDashboardRecommendationCode || isLoading) {
			return;
		}

		const normalizedCode = pendingDashboardRecommendationCode.trim().toLowerCase();
		if (!normalizedCode) {
			setPendingDashboardRecommendationCode(null);
			return;
		}

		const targetItem = filteredAndSortedItems.find(
			(item) => String(item.kodZalecenia ?? "").trim().toLowerCase() === normalizedCode,
		);

		if (!targetItem) {
			return;
		}

		const rowIndex = filteredAndSortedItems.findIndex((item) => item.id === targetItem.id);
		if (rowIndex < 0) {
			return;
		}

		const targetPage = Math.floor(rowIndex / pageSize) + 1;
		handlePageChange(targetPage);
		setSelectedId(targetItem.id);
		setPendingDashboardRecommendationCode(null);

		if (typeof window !== "undefined") {
			window.sessionStorage.removeItem(DASHBOARD_OPEN_RECOMMENDATION_CODE_KEY);
		}
	}, [
		filteredAndSortedItems,
		handlePageChange,
		isLoading,
		pageSize,
		pendingDashboardRecommendationCode,
	]);

	const visibleRecommendationColumnDefinitions = useMemo(
		() =>
			RECOMMENDATION_COLUMNS.filter((column) =>
				visibleRecommendationColumns.includes(column.key),
			),
		[visibleRecommendationColumns],
	);

	const draftSelectableColumnDefinitions = RECOMMENDATION_COLUMNS;

	const loadItems = async () => {
		setError(null);
		setIsLoading(true);

		const result = await fetchRecommendations(operatorLogin, {
			sortBy: "id",
			sortOrder: "asc",
		});

		if (!result.ok) {
			setItems([]);
			setTotal(0);
			setError(result.error);
			setIsLoading(false);
			return;
		}

		setItems(result.data.items);
		setTotal(result.data.total);
		setSelectedId((prev) =>
			prev && result.data.items.some((item) => item.id === prev) ? prev : null,
		);
		setIsLoading(false);
	};

	const loadInspectionOptions = async () => {
		setIsInspectionOptionsLoading(true);

		try {
			const response = await fetch(AVAILABLE_INSPECTIONS_API_URL, {
				method: "GET",
				headers: {
					"Content-Type": "application/json",
					"X-Operator-Login": operatorLogin,
				},
				cache: "no-store",
			});

			if (!response.ok) {
				setInspectionOptions([]);
				if (response.status === 401) {
					setFormError(
						"Brak autoryzacji operatora. Odśwież sesję i zaloguj się ponownie.",
					);
				} else if (response.status === 403) {
					setFormError("Brak uprawnień do listy dostępnych inspekcji.");
				}
				return;
			}

			const payload = (await response.json()) as
				| Array<{
						id?: unknown;
						lp?: unknown;
						inspectionKod?: unknown;
						kodInspekcji?: unknown;
						nazwaPodmiotu?: unknown;
				  }>
				| {
						items?: Array<{
							id?: unknown;
							lp?: unknown;
							inspectionKod?: unknown;
							kodInspekcji?: unknown;
							nazwaPodmiotu?: unknown;
						}>;
				  };
			const rawItems = Array.isArray(payload) ? payload : (payload.items ?? []);

			const mapped = rawItems
				.map((item) => {
					const id = Number(item.id);
					const lp = Number(item.lp);
					const inspectionCode = String(
						(item as { inspectionKod?: unknown }).inspectionKod ??
							(item as { kodInspekcji?: unknown }).kodInspekcji ??
							(item as { inspectionCode?: unknown }).inspectionCode ??
							item.lp ??
							"",
					).trim();
					const nazwaPodmiotu = String(item.nazwaPodmiotu ?? "").trim();

					if (!Number.isFinite(id) || id <= 0 || !inspectionCode) {
						return null;
					}

					return {
						id,
						lp: Number.isFinite(lp) && lp > 0 ? lp : id,
						inspectionCode,
						nazwaPodmiotu,
					};
				})
				.filter((item): item is InspectionOption => Boolean(item))
				.sort((left, right) =>
					left.inspectionCode.localeCompare(right.inspectionCode, "pl", {
						numeric: true,
						sensitivity: "base",
					}),
				);

			setInspectionOptions(mapped);
		} catch {
			setInspectionOptions([]);
			setFormError("Nie udało się pobrać dostępnych inspekcji.");
		} finally {
			setIsInspectionOptionsLoading(false);
		}
	};

	const loadRecommendationStatusOptions = async () => {
		try {
			const result = await fetchDictionaryEntries("statusy_zalecen");
			if (!result.ok) {
				setRecommendationStatusOptions([]);
				return;
			}

			setRecommendationStatusOptions(
				mapDictionaryEntriesToOptions(result.data),
			);
		} catch {
			setRecommendationStatusOptions([]);
		}
	};

	const loadEntityNameOptions = async () => {
		try {
			const result = await fetchDictionaryEntries("nazwy_podmiotow");
			if (!result.ok) {
				setEntityNameOptions([]);
				return;
			}

			setEntityNameOptions(mapDictionaryEntriesToOptions(result.data));
		} catch {
			setEntityNameOptions([]);
		}
	};

	useEffect(() => {
		void loadItems();
		void loadRecommendationStatusOptions();
		void loadEntityNameOptions();
	}, []);

	useEffect(() => {
		void loadInspectionOptions();
	}, [operatorLogin]);

	useEffect(() => {
		const handleInspectionsChanged = () => {
			void loadInspectionOptions();
		};

		window.addEventListener(INSPECTIONS_CHANGED_EVENT, handleInspectionsChanged);
		return () => {
			window.removeEventListener(INSPECTIONS_CHANGED_EVENT, handleInspectionsChanged);
		};
	}, [operatorLogin]);

	useEffect(() => {
		if (form.isInspectionMissing) {
			return;
		}

		if (!selectedInspectionOption) {
			setForm((prev) => ({
				...prev,
				nazwaPodmiotu: "",
			}));
			return;
		}

		setForm((prev) => ({
			...prev,
			nazwaPodmiotu: selectedInspectionOption.nazwaPodmiotu,
		}));
	}, [form.isInspectionMissing, selectedInspectionOption]);

	const openAdvancedFilterForColumn = (
		columnKey: RecommendationColumnKey,
		triggerElement: HTMLElement,
	) => {
		setAdvancedFilterAnchor(getFloatingPanelAnchor(triggerElement));
		setAdvancedFilterColumnKey(columnKey);
		setAdvancedFilterSearch("");
		setIsAdvancedFilterModalOpen(true);
	};

	const handleExportCurrentView = async (
		inspectionColumnKeys: InspectionExportColumnKey[],
		sanctionColumnKeys: SanctionExportColumnKey[],
		decisionColumnKeys: DecisionExportColumnKey[],
		includeInspections: boolean,
		includeSanctions: boolean,
		includeDecisions: boolean,
	) => {
		if (
			isExporting ||
			filteredAndSortedItems.length === 0 ||
			visibleRecommendationColumnDefinitions.length === 0
		) {
			return;
		}

		setIsExporting(true);
		setError(null);

		try {
			const workbook = await createStyledExportWorkbook("Ewidencja zaleceń");

			const linkedInspectionIds = new Set(
				filteredAndSortedItems
					.map((item) => item.inspectionId)
					.filter(
						(value): value is number =>
							typeof value === "number" && Number.isFinite(value) && value > 0,
					),
			);

			const loadInspectionLpMap = async (url: string) => {
				try {
					const response = await fetch(url, {
						method: "GET",
						headers: {
							"Content-Type": "application/json",
							"X-Operator-Login": operatorLogin,
						},
						cache: "no-store",
					});

					if (!response.ok) {
						return new Map<number, number>();
					}

					const payload = (await response.json()) as
						| Array<{ id?: unknown; lp?: unknown }>
						| { items?: Array<{ id?: unknown; lp?: unknown }> };
					const rawItems = Array.isArray(payload)
						? payload
						: (payload.items ?? []);

					return new Map(
						rawItems
							.map((item) => {
								const id = Number(item.id);
								const lp = Number(item.lp);
								if (
									!Number.isFinite(id) ||
									id <= 0 ||
									!Number.isFinite(lp) ||
									lp <= 0
								) {
									return null;
								}

								return [id, lp] as const;
							})
							.filter((entry): entry is readonly [number, number] => entry !== null),
					);
				} catch {
					return new Map<number, number>();
				}
			};

			const [inspectionsResponse, sanctionsResult, decisionsResult, sanctionsLpById] =
				await Promise.all([
					fetch(INSPECTIONS_API_URL, {
						method: "GET",
						headers: {
							"Content-Type": "application/json",
							"X-Operator-Login": operatorLogin,
						},
						cache: "no-store",
					}),
					fetchSanctionRequests(operatorLogin, {
						sortBy: "id",
						sortOrder: "asc",
					}),
					fetchObligatingDecisions(operatorLogin),
					loadInspectionLpMap(INSPECTIONS_API_URL),
				]);

			const rawInspectionRows: unknown[] = [];

			if (inspectionsResponse.ok) {
				const payload = (await inspectionsResponse.json()) as
					| unknown[]
					| { items?: unknown[] };
				const items = Array.isArray(payload) ? payload : (payload.items ?? []);
				rawInspectionRows.push(...items);
			}

			const mappedInspections = rawInspectionRows.map((rawRow, index) =>
				normalizeInspectionRow((rawRow ?? {}) as RawInspectionRow, index),
			);

			const relatedInspections = mappedInspections.filter((row) =>
				linkedInspectionIds.has(Number(row.id)),
			);

			const relatedSanctionsSource = sanctionsResult.ok
				? sanctionsResult.data.items
				: [];
			const relatedDecisionsSource = decisionsResult.ok
				? decisionsResult.data.items
				: [];
			const relatedSanctions = relatedSanctionsSource.filter(
				(item) =>
					typeof item.inspectionId === "number" &&
					linkedInspectionIds.has(item.inspectionId),
			);

			const inspectionCodeByIdForExport = new Map(
				relatedInspections.map((row) => [Number(row.id), row.kodInspekcji]),
			);

			const inspectionCodeByRecommendationCode = new Map<string, string>();
			for (const recommendation of filteredAndSortedItems) {
				const recommendationCode = String(recommendation.kodZalecenia ?? "")
					.trim()
					.toUpperCase();
				if (!recommendationCode) {
					continue;
				}

				const inspectionId = recommendation.inspectionId ?? null;
				const inspectionCode =
					resolveInspectionCode({
						inspectionKod: recommendation.inspectionKod,
						kodInspekcji: recommendation.kodInspekcji,
						inspectionLp: recommendation.inspectionLp,
						inspectionId,
					}) ||
					(typeof inspectionId === "number"
						? String(
								inspectionCodeByIdForExport.get(inspectionId) ??
								sanctionsLpById.get(inspectionId) ??
								"",
							)
						: "");

				inspectionCodeByRecommendationCode.set(recommendationCode, inspectionCode);
			}

			const relatedDecisions = relatedDecisionsSource.filter((item) => {
				const recommendationCode = String(item.recommendationKodZalecenia ?? "")
					.trim()
					.toUpperCase();
				return recommendationCode.length > 0 &&
					inspectionCodeByRecommendationCode.has(recommendationCode);
			});

			const recommendationHeaders = visibleRecommendationColumnDefinitions.map(
				(column) => column.label,
			);
			const recommendationRows = filteredAndSortedItems.map((item) =>
				visibleRecommendationColumnDefinitions.map((column) =>
					getCellValue(item, column.key),
				),
			);

			addWorksheetWithStyles(
				workbook,
				"Zalecenia",
				recommendationHeaders,
				recommendationRows,
			);

			if (includeInspections && inspectionColumnKeys.length > 0) {
				const inspectionHeaders = inspectionColumnKeys.map(
					(key) =>
						INSPECTION_EXPORT_COLUMNS.find((column) => column.key === key)?.label ??
						key,
				);
				const inspectionRowsForExport = relatedInspections.map((row) =>
					inspectionColumnKeys.map((key) => String(row[key] ?? "")),
				);
				addWorksheetWithStyles(
					workbook,
					"Inspekcje",
					inspectionHeaders,
					inspectionRowsForExport,
				);
			}

			if (includeSanctions && sanctionColumnKeys.length > 0) {
				const sanctionHeaders = sanctionColumnKeys.map(
					(key) =>
						SANCTION_EXPORT_COLUMNS.find((column) => column.key === key)?.label ??
						key,
				);
				const sanctionRowsForExport = relatedSanctions.map((item) => {
					const inspectionId = item.inspectionId ?? null;
					const inspectionCode =
						resolveInspectionCode({
							inspectionKod: item.inspectionKod,
							kodInspekcji: item.kodInspekcji,
							inspectionLp: item.inspectionLp,
							inspectionId,
						}) ||
						(typeof inspectionId === "number"
							? String(
									inspectionCodeByIdForExport.get(inspectionId) ??
										sanctionsLpById.get(inspectionId) ??
										"",
								)
							: "");

					return sanctionColumnKeys.map((key) => {
						switch (key) {
							case "lp":
								return String(item.lp);
							case "requestId":
								return String(item.kodSankcji ?? item.lp ?? "").trim();
							case "inspectionLp":
								return inspectionCode;
							case "nazwaPodmiotuObjetegoInspekcja":
								return item.nazwaPodmiotuObjetegoInspekcja ?? "";
							case "nazwaPodmiotuObjetegoSankcjaList":
								return item.nazwaPodmiotuObjetegoSankcjaList.join(", ");
							case "dataWniosku":
								return item.dataWniosku ?? "";
							case "wniosekDo":
								return item.wniosekDo ?? "";
							case "sankcjaList":
								return item.sankcjaList.join(", ");
							case "podstawaPrawnaSankcjiList":
								return item.podstawaPrawnaSankcjiList.join(", ");
							case "naruszeniaSkutkujaceSankcjaList":
								return item.naruszeniaSkutkujaceSankcjaList.join(", ");
							case "czyMamyInformacjeOWszczeciuPostepowania":
								return item.czyMamyInformacjeOWszczeciuPostepowania ?? "";
							case "rozstrzygniecie":
								return item.rozstrzygniecie ?? "";
							case "komentarz":
								return item.komentarz ?? "";
						}
					});
				});
				addWorksheetWithStyles(
					workbook,
					"Wnioski sankcyjne",
					sanctionHeaders,
					sanctionRowsForExport,
				);
			}

			if (includeDecisions && decisionColumnKeys.length > 0) {
				const decisionHeaders = decisionColumnKeys.map(
					(key) =>
						DECISION_EXPORT_COLUMNS.find((column) => column.key === key)?.label ??
						key,
				);
				const decisionRowsForExport = relatedDecisions.map((item, index) =>
					decisionColumnKeys.map((key) => {
						const recommendationCode = String(
							item.recommendationKodZalecenia ?? "",
						)
							.trim()
							.toUpperCase();
						switch (key) {
							case "lp":
								return String(index + 1);
							case "kodDecyzji":
								return item.kodDecyzji ?? "";
							case "kodZalecenia":
								return recommendationCode;
							case "inspectionLp":
								return (
									inspectionCodeByRecommendationCode.get(recommendationCode) ?? ""
								);
							case "nazwaPodmiotu":
								return item.nazwaPodmiotu ?? "";
							case "liczbaZalecen":
								return item.liczbaZalecen === null ? "" : String(item.liczbaZalecen);
							case "dataWszczeciaPostepowaniaIInstancji":
								return item.dataWszczeciaPostepowaniaIInstancji ?? "";
							case "osobyProwadzaceIInstancjeList":
								return (item.osobyProwadzaceIInstancjeList ?? []).join(", ");
							case "dataDecyzjiIInstancji":
								return item.dataDecyzjiIInstancji ?? "";
							case "dataDoreczeniaDecyzjiIInstancji":
								return item.dataDoreczeniaDecyzjiIInstancji ?? "";
							case "rozstrzygniecieI":
								return item.rozstrzygniecieI ?? "";
							case "dataWnioskuPonowneRozpatrzenie":
								return item.dataWnioskuPonowneRozpatrzenie ?? "";
							case "dataWplywuWnioskuPonowneRozpatrzenie":
								return item.dataWplywuWnioskuPonowneRozpatrzenie ?? "";
							case "osobyProwadzaceIIInstancjeList":
								return (item.osobyProwadzaceIIInstancjeList ?? []).join(", ");
							case "dataDecyzjiIIInstancji":
								return item.dataDecyzjiIIInstancji ?? "";
							case "dataDoreczeniaDecyzjiIIInstancji":
								return item.dataDoreczeniaDecyzjiIIInstancji ?? "";
							case "rozstrzygniecieII":
								return item.rozstrzygniecieII ?? "";
							case "komentarz":
								return item.komentarz ?? "";
						}
					}),
				);
				addWorksheetWithStyles(
					workbook,
					"Decyzje zobowiązujące",
					decisionHeaders,
					decisionRowsForExport,
				);
			}

			const fileName = "zalecenia-inspekcje-sankcje-decyzje.xlsx";
			await saveWorkbookAsXlsx(workbook, fileName);
		} catch (caughtError) {
			if (
				caughtError instanceof DOMException &&
				caughtError.name === "AbortError"
			) {
				return;
			}

			setError("Nie udało się wyeksportować danych do Excela.");
		} finally {
			setIsExporting(false);
		}
	};

	const handleOpenExportConfigModal = () => {
		if (isExporting || filteredAndSortedItems.length === 0) {
			return;
		}

		setIncludeInspectionsInExport(false);
		setIncludeSanctionsInExport(false);
		setIncludeDecisionsInExport(false);
		setActiveExportColumnsTab("inspections");
		setIsExportConfigModalOpen(true);
	};

	const toggleInspectionExportColumn = (
		columnKey: InspectionExportColumnKey,
		isSelected: boolean,
	) => {
		setSelectedInspectionExportColumns((prev) => {
			const nextSet = new Set(prev);
			if (isSelected) {
				nextSet.add(columnKey);
			} else {
				if (prev.length <= 1) {
					return prev;
				}
				nextSet.delete(columnKey);
			}

			return INSPECTION_EXPORT_COLUMNS.map((column) => column.key).filter((key) =>
				nextSet.has(key),
			);
		});
	};

	const toggleSanctionExportColumn = (
		columnKey: SanctionExportColumnKey,
		isSelected: boolean,
	) => {
		setSelectedSanctionExportColumns((prev) => {
			const nextSet = new Set(prev);
			if (isSelected) {
				nextSet.add(columnKey);
			} else {
				if (prev.length <= 1) {
					return prev;
				}
				nextSet.delete(columnKey);
			}

			return SANCTION_EXPORT_COLUMNS.map((column) => column.key).filter((key) =>
				nextSet.has(key),
			);
		});
	};

	const toggleDecisionExportColumn = (
		columnKey: DecisionExportColumnKey,
		isSelected: boolean,
	) => {
		setSelectedDecisionExportColumns((prev) => {
			const nextSet = new Set(prev);
			if (isSelected) {
				nextSet.add(columnKey);
			} else {
				if (prev.length <= 1) {
					return prev;
				}
				nextSet.delete(columnKey);
			}

			return DECISION_EXPORT_COLUMNS.map((column) => column.key).filter((key) =>
				nextSet.has(key),
			);
		});
	};

	const handleConfirmExportFromModal = () => {
		if (
			(includeInspectionsInExport && selectedInspectionExportColumns.length === 0) ||
			(includeSanctionsInExport && selectedSanctionExportColumns.length === 0) ||
			(includeDecisionsInExport && selectedDecisionExportColumns.length === 0)
		) {
			return;
		}

		const orderedInspectionColumns = INSPECTION_EXPORT_COLUMNS.map(
			(column) => column.key,
		).filter((key) => selectedInspectionExportColumns.includes(key));

		const orderedSanctionColumns = SANCTION_EXPORT_COLUMNS.map(
			(column) => column.key,
		).filter((key) => selectedSanctionExportColumns.includes(key));

		const orderedDecisionColumns = DECISION_EXPORT_COLUMNS.map(
			(column) => column.key,
		).filter((key) => selectedDecisionExportColumns.includes(key));

		setIsExportConfigModalOpen(false);
		void handleExportCurrentView(
			orderedInspectionColumns,
			orderedSanctionColumns,
			orderedDecisionColumns,
			includeInspectionsInExport,
			includeSanctionsInExport,
			includeDecisionsInExport,
		);
	};

	const openCreateModal = async () => {
		if (!canManageRecommendations) {
			setError("Konto zewnętrzne ma dostęp tylko do odczytu.");
			return;
		}

		setEditingItem(null);
		setForm(EMPTY_FORM);
		setFormError(null);
		setShowRequiredFieldErrors(false);
		setVersionConflictUpdatedAt(null);
		setSaveLockConflict(null);
		setIsFormOpen(true);
		await loadInspectionOptions();
	};

	const openEditModal = async () => {
		if (!canManageRecommendations) {
			setError("Konto zewnętrzne ma dostęp tylko do odczytu.");
			return;
		}

		if (!selectedItem || !selectedItem.canEdit) {
			return;
		}

		setEditingItem(selectedItem);
		setForm(recommendationToForm(selectedItem));
		setFormError(null);
		setShowRequiredFieldErrors(false);
		setVersionConflictUpdatedAt(null);
		setSaveLockConflict(null);
		await loadInspectionOptions();
		setIsFormOpen(true);
	};

	const closeModal = () => {
		if (editRecordLock.lockToken) {
			void editRecordLock.release();
		}

		setIsFormOpen(false);
		setEditingItem(null);
		setFormError(null);
		setShowRequiredFieldErrors(false);
		setVersionConflictUpdatedAt(null);
		setSaveLockConflict(null);
		setIsSubmitting(false);
	};
	closeModalRef.current = closeModal;

	const handleRefreshAfterConflict = async () => {
		if (!editingItem) {
			return;
		}

		const result = await fetchRecommendations(operatorLogin, {
			sortBy: "id",
			sortOrder: "asc",
		});

		if (!result.ok) {
			setFormError(result.error);
			return;
		}

		setItems(result.data.items);
		setTotal(result.data.total);
		const refreshed = result.data.items.find((item) => item.id === editingItem.id);
		if (!refreshed) {
			closeModal();
			return;
		}

		setEditingItem(refreshed);
		setForm(recommendationToForm(refreshed));
		setFormError(null);
		setVersionConflictUpdatedAt(null);
		setSaveLockConflict(null);
	};

	const openDeleteModal = () => {
		if (!isDirector || !selectedItem) {
			return;
		}

		setError(null);
		setIsDeleteConfirmModalOpen(true);
	};

	const handleDeleteItem = async () => {
		if (!isDirector || !selectedItem || isDeletingItem) {
			return;
		}

		const deletedEntityName = selectedItem.nazwaPodmiotu?.trim() ?? "";

		setIsDeletingItem(true);
		setError(null);

		const result = await deleteRecommendation(operatorLogin, selectedItem.id);
		if (!result.ok) {
			setError(result.error);
			setIsDeletingItem(false);
			return;
		}

		setIsDeleteConfirmModalOpen(false);
		setSelectedId(null);
		await loadItems();
		window.dispatchEvent(new CustomEvent(RECOMMENDATIONS_CHANGED_EVENT));
		setDeleteSuccessEntityName(deletedEntityName);
		setIsDeleteSuccessModalOpen(true);
		setIsDeletingItem(false);
	};

	const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
		event.preventDefault();
		if (!canManageRecommendations) {
			setFormError("Konto zewnętrzne ma dostęp tylko do odczytu.");
			return;
		}

		if (shouldShowLockedByOtherUser) {
			setFormError(
				"Nie możesz teraz edytować tego wpisu, ponieważ jest edytowany przez innego użytkownika.",
			);
			return;
		}

		const wasEditing = Boolean(editingItem);
		const isRequiredInspectionMissing =
			!form.isInspectionMissing && !form.inspectionId.trim();
		const isRequiredEntityNameMissing =
			form.isInspectionMissing && !form.nazwaPodmiotu.trim();
		const isRequiredPositionMissing = !form.pozycja.trim();
		const hasMissingRequiredFields =
			isRequiredInspectionMissing ||
			isRequiredEntityNameMissing ||
			isRequiredPositionMissing;

		setShowRequiredFieldErrors(true);

		if (hasMissingRequiredFields) {
			setFormError(null);
			return;
		}

		const payload = formToPayload(form);
		if (!payload) {
			setFormError(
				"Wprowadź poprawne wartości: id inspekcji i liczba zaleceń muszą być poprawne.",
			);
			return;
		}

		if (!payload.nazwaPodmiotu.trim()) {
			setFormError(null);
			return;
		}

		setShowRequiredFieldErrors(false);

		if (editingItem) {
			const basePayload = formToPayload(recommendationToForm(editingItem));
			if (basePayload && JSON.stringify(payload) === JSON.stringify(basePayload)) {
				setFormError("Brak zmian do zapisania.");
				return;
			}
		}

		setIsSubmitting(true);
		setFormError(null);
		setVersionConflictUpdatedAt(null);
		setSaveLockConflict(null);

		try {
			const result = editingItem
				? await updateRecommendation(operatorLogin, editingItem.id, payload, {
						expectedUpdatedAt: editingItem.zaktualizowanoO,
						lockToken: editRecordLock.lockToken,
				  })
				: await createRecommendation(operatorLogin, payload);

			if (!result.ok) {
				if (result.status === 423) {
					if (result.lockErrorCode === "RECORD_LOCKED") {
						setSaveLockConflict(result.lockConflict ?? null);
						setFormError(
							"Nie możesz teraz edytować tego wpisu, ponieważ jest edytowany przez innego użytkownika.",
						);
						return;
					}

					setSaveLockConflict(null);
					setFormError(result.error);
					return;
				}

				if (result.status === 409) {
					if (wasEditing) {
						setVersionConflictUpdatedAt(result.currentUpdatedAt ?? null);
						setFormError(
							"Dane zostały zmienione przez innego użytkownika. Odśwież widok i spróbuj ponownie.",
						);
					} else {
						setFormError(result.error);
					}
					return;
				}

				setFormError(result.error);
				return;
			}

			closeModal();
			setSelectedId(result.data.id);
			setSuccessEntityName(payload.nazwaPodmiotu.trim());
			setSuccessInspectionCode(
				resolveInspectionCode({
					inspectionKod: result.data.inspectionKod,
					kodInspekcji: result.data.kodInspekcji,
					inspectionLp: result.data.inspectionLp,
					inspectionId: result.data.inspectionId,
				}) ||
				(selectedInspectionOption?.inspectionCode ?? ""),
			);
			setSuccessMode(wasEditing ? "edit" : "create");
			setIsSuccessModalOpen(true);
			void loadItems();
			window.dispatchEvent(new CustomEvent(RECOMMENDATIONS_CHANGED_EVENT));
		} catch {
			setFormError("Nie udało się zapisać zalecenia.");
		} finally {
			setIsSubmitting(false);
		}
	};

	const isRequiredInspectionMissing =
		showRequiredFieldErrors && !form.isInspectionMissing && !form.inspectionId.trim();
	const isRequiredEntityNameMissing =
		showRequiredFieldErrors && form.isInspectionMissing && !form.nazwaPodmiotu.trim();
	const isRequiredPositionMissing =
		showRequiredFieldErrors && !form.pozycja.trim();

	return (
		<section className="rounded-2xl border border-slate-700/70 bg-[#101f39] p-4 sm:p-5">
			<TablePanelToolbar
				title="Zalecenia"
				canClearFilters={canClearFilters}
				isExporting={isExporting}
				hasRowsToExport={
					filteredAndSortedItems.length > 0 &&
					visibleRecommendationColumnDefinitions.length > 0
				}
				onOpenViewModal={handleOpenViewModal}
				onClearFilters={clearFilters}
				onExport={handleOpenExportConfigModal}
				actions={
					<>
						{canManageRecommendations ? (
							<>
								<button
									type="button"
									onClick={() => void openCreateModal()}
									className="inline-flex h-10 items-center gap-2 rounded-lg border border-[#8ec5a1] bg-[#b9e8c9] px-3.5 font-semibold text-[#1f5130] text-sm transition-colors hover:bg-[#a5debb]"
								>
									<Plus size={15} />
									Dodaj zalecenie
								</button>

								<button
									type="button"
									onClick={() => void openEditModal()}
									disabled={!selectedItem || !selectedItem.canEdit}
									className="inline-flex h-10 items-center gap-2 rounded-lg border px-3.5 font-semibold text-sm transition-colors enabled:border-[#7ea8e7] enabled:bg-[#c7dcff] enabled:text-[#1d4882] enabled:hover:bg-[#b7d3ff] disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-[#1a2946] disabled:text-slate-500"
								>
									<Pencil size={15} />
									Edytuj
								</button>
							</>
						) : null}

						{isDirector ? (
							<button
								type="button"
								onClick={openDeleteModal}
								disabled={!selectedItem || isDeletingItem}
								className="inline-flex h-10 items-center gap-2 rounded-lg border border-[#f2a3a3] bg-[#6f2a36] px-3.5 font-semibold text-[#ffe5e8] text-sm transition-colors hover:bg-[#833242] disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-[#1a2946] disabled:text-slate-500"
							>
								<Trash2 size={15} />
								Usuń
							</button>
						) : null}
					</>
				}
			/>

			{error ? (
				<p className="mb-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 font-medium text-rose-700 text-sm">
					{error}
				</p>
			) : null}

			<TableSurface
				isLoading={isLoading}
				containerClassName="-mt-1"
				scrollAreaClassName="h-[calc(96vh-11rem)] min-h-88"
				footer={
					<TablePagination
						currentPage={currentPage}
						totalPages={totalPages}
						paginationItems={paginationItems}
						totalItems={filteredAndSortedItems.length}
						pageSize={pageSize}
						onPageChange={handlePageChange}
						pageSizeOptions={[...TABLE_PAGE_SIZE_OPTIONS]}
						onPageSizeChange={handlePageSizeChange}
						showWhenSinglePage
					/>
				}
			>
				<table className="min-w-350 border-collapse text-slate-900 text-sm">
					<TableHeaderWithFilters
						visibleColumns={visibleRecommendationColumnDefinitions.map((column) => ({
							...column,
							tooltip: RECOMMENDATION_COLUMN_TOOLTIPS[column.key],
						}))}
						sortColumnKey={sortColumnKey}
						sortDirection={sortDirection}
						advancedFilters={advancedFilters}
						columnFilters={columnFilters}
						onSortByColumn={handleSortByColumn}
						onOpenAdvancedFilter={openAdvancedFilterForColumn}
						onFilterChange={handleFilterChange}
					/>
					<tbody>
						{paginatedRecommendationItems.map((item) => {
							const isSelected = selectedId === item.id;
							return (
								<tr
									key={item.id}
									onClick={() => setSelectedId(item.id)}
									className={`cursor-pointer border-slate-200 border-b transition-colors last:border-b-0 ${
										isSelected
											? "bg-blue-100 text-slate-900 ring-1 ring-blue-300 ring-inset"
											: "bg-white text-slate-900 hover:bg-slate-50"
									}`}
								>
									{visibleRecommendationColumnDefinitions.map((column) => {
										const rawValue = getCellValue(item, column.key);
										const normalizedRawValue = rawValue.trim();
										const value =
											column.key === "inspectionId"
												? normalizedRawValue || "Brak powiązania"
												: column.key === "status"
													? normalizedRawValue.toLowerCase() === "brak"
														? "-"
														: normalizedRawValue || "-"
													: normalizedRawValue || "-";
										const isScrollableValue =
											column.key === "komentarz" ||
											column.key === "dataZalecenList" ||
											column.key === "dataAkceptacjiNotyWeryfikacjiList";

										return (
											<td
												key={column.key}
												className="whitespace-nowrap px-3 py-2.5 font-normal"
											>
												{isScrollableValue ? (
													<div className="subtle-horizontal-scroll max-w-64 overflow-x-auto whitespace-nowrap pr-1">
														{value}
													</div>
												) : (
													value
												)}
											</td>
										);
									})}
								</tr>
							);
						})}
						{!isLoading && filteredAndSortedItems.length === 0 ? (
							<tr>
								<td
									colSpan={visibleRecommendationColumnDefinitions.length}
									className="px-3 py-6 text-center text-slate-500 text-sm"
								>
									Brak rekordów. Łącznie: {total}.
								</td>
							</tr>
						) : null}
					</tbody>
				</table>
			</TableSurface>

			<ExportConfigModal
				isOpen={isExportConfigModalOpen}
				description="Zalecenia eksportują aktualny widok tabeli. Wybierz dane powiązane."
				relationsLabel="Powiąż wybrane zalecenia z:"
				relations={[
					{
						id: "inspections",
						label: "Inspekcjami",
						enabled: includeInspectionsInExport,
						selectedCount: selectedInspectionExportColumns.length,
						onToggle: () => {
							setIncludeInspectionsInExport((prev) => {
								const next = !prev;
								if (next) {
									setActiveExportColumnsTab("inspections");
								}
								return next;
							});
						},
					},
					{
						id: "sanctions",
						label: "Wnioskami sankcyjnymi",
						enabled: includeSanctionsInExport,
						selectedCount: selectedSanctionExportColumns.length,
						onToggle: () => {
							setIncludeSanctionsInExport((prev) => {
								const next = !prev;
								if (next) {
									setActiveExportColumnsTab("sanctions");
								}
								return next;
							});
						},
					},
					{
						id: "decisions",
						label: "Decyzjami zobowiązującymi",
						enabled: includeDecisionsInExport,
						selectedCount: selectedDecisionExportColumns.length,
						onToggle: () => {
							setIncludeDecisionsInExport((prev) => {
								const next = !prev;
								if (next) {
									setActiveExportColumnsTab("decisions");
								}
								return next;
							});
						},
					},
				]}
				tabs={[
					{
						id: "inspections",
						label: "Inspekcje",
						columns: INSPECTION_EXPORT_COLUMNS.map((column) => ({
							key: column.key,
							label: column.label,
						})),
						selectedKeys: selectedInspectionExportColumns,
						onToggleKey: (key, isSelected) =>
							toggleInspectionExportColumn(
								key as InspectionExportColumnKey,
								isSelected,
							),
						onSelectAll: () =>
							setSelectedInspectionExportColumns(
								INSPECTION_EXPORT_COLUMNS.map((column) => column.key),
							),
					},
					{
						id: "sanctions",
						label: "Wnioski sankcyjne",
						columns: SANCTION_EXPORT_COLUMNS.map((column) => ({
							key: column.key,
							label: column.label,
						})),
						selectedKeys: selectedSanctionExportColumns,
						onToggleKey: (key, isSelected) =>
							toggleSanctionExportColumn(key as SanctionExportColumnKey, isSelected),
						onSelectAll: () =>
							setSelectedSanctionExportColumns(
								SANCTION_EXPORT_COLUMNS.map((column) => column.key),
							),
					},
					{
						id: "decisions",
						label: "Decyzje zobowiązujące",
						columns: DECISION_EXPORT_COLUMNS.map((column) => ({
							key: column.key,
							label: column.label,
						})),
						selectedKeys: selectedDecisionExportColumns,
						onToggleKey: (key, isSelected) =>
							toggleDecisionExportColumn(key as DecisionExportColumnKey, isSelected),
						onSelectAll: () =>
							setSelectedDecisionExportColumns(
								DECISION_EXPORT_COLUMNS.map((column) => column.key),
							),
					},
				]}
				activeTabId={activeExportColumnsTab}
				onActiveTabChange={(tabId) =>
					setActiveExportColumnsTab(tabId as "inspections" | "sanctions" | "decisions")
				}
				onClose={() => setIsExportConfigModalOpen(false)}
				onConfirm={handleConfirmExportFromModal}
				isConfirmDisabled={
					isExporting ||
					(includeInspectionsInExport && selectedInspectionExportColumns.length === 0) ||
					(includeSanctionsInExport && selectedSanctionExportColumns.length === 0) ||
					(includeDecisionsInExport && selectedDecisionExportColumns.length === 0)
				}
				isExporting={isExporting}
			/>

			<TableAdvancedFilterModal
				isOpen={isAdvancedFilterModalOpen}
				anchor={advancedFilterAnchor}
				columnLabel={
					RECOMMENDATION_COLUMNS.find(
						(column) => column.key === advancedFilterColumnKey,
					)?.label ?? "Kolumna"
				}
				searchValue={advancedFilterSearch}
				visibleValues={visibleAdvancedFilterValues}
				selectedValues={selectedAdvancedFilterValues}
				onClose={() => setIsAdvancedFilterModalOpen(false)}
				onSearchChange={setAdvancedFilterSearch}
				onSelectAllVisible={selectAllVisibleAdvancedFilterValues}
				onClearSelectedColumn={clearAdvancedFilterForSelectedColumn}
				onToggleValue={toggleAdvancedFilterValue}
				onClearAllFilters={clearFilters}
			/>

			<TableColumnPickerModal<RecommendationColumnKey, never>
				isOpen={isColumnPickerOpen}
				columns={draftSelectableColumnDefinitions}
				hiddenColumns={draftHiddenColumns}
				visibleColumnsCount={draftVisibleRecommendationColumns.length}
				onClose={() => setIsColumnPickerOpen(false)}
				onChangeColumnVisibility={handleDraftColumnVisibilityChange}
				onShowAllColumns={() => setDraftHiddenColumns([])}
				onApply={handleApplyViewChanges}
			/>

			<RegistryFormScaffold
				isOpen={isFormOpen}
				title={editingItem ? "Edytuj zalecenie" : "Dodaj zalecenie"}
				subtitle={editingItem ? `Id zalecenia: ${editingItem.kodZalecenia}` : undefined}
				onClose={closeModal}
				onSubmit={(event) => void handleSubmit(event)}
				isContentReadOnly={isReadOnlyDueToLock}
				closeOnBackdropClick={false}
				headerNotices={
					<>
						{inactivityTimeout.isWarning ? (
							<div className="mt-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-amber-900 text-sm">
								<p className="font-semibold">
									Nie wykryto aktywności. Formularz zostanie zamknięty za{" "}
									<span className="tabular-nums">{inactivityTimeout.secondsRemaining}</span> s.
								</p>
								<button
									type="button"
									onClick={inactivityTimeout.resetTimer}
									className="mt-2 inline-flex h-7 items-center rounded border border-amber-400 bg-amber-100 px-2 font-semibold text-amber-900 text-xs transition-colors hover:bg-amber-200"
								>
									Kontynuuj edycję
								</button>
							</div>
						) : null}

						{isEditMode && shouldShowLockedByOtherUser ? (
							<div className="mt-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-rose-800 text-sm">
								<p className="font-semibold">
									Nie możesz teraz edytować tego wpisu, ponieważ jest edytowany przez innego użytkownika.
								</p>
								<p className="mt-1">
									Rekord edytuje teraz: {lockOwnerLabel}, od {formatLockStartHourMinute(lockAcquiredAt)}.
								</p>
							</div>
						) : null}

						{isEditMode && editRecordLock.isConnectionLost ? (
							<p className="mt-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 font-medium text-amber-800 text-sm">
								{editRecordLock.error ?? "Utracono połączenie z serwerem — trwa próba odnowienia blokady..."}
							</p>
						) : null}

						{isEditMode && editRecordLock.isExpired ? (
							<p className="mt-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 font-medium text-rose-800 text-sm">
								{editRecordLock.error ?? "Czas edycji wygasł — połączenie zostało przerwane zbyt długo. Zamknij formularz i otwórz ponownie."}
							</p>
						) : null}

						{isEditMode && editRecordLock.isAcquireFailed ? (
							<div className="mt-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-rose-800 text-sm">
								<p className="font-medium">
									{editRecordLock.error ?? "Nie udało się założyć blokady rekordu."}
								</p>
								<button
									type="button"
									onClick={() => editRecordLock.retryAcquire()}
									className="mt-2 inline-flex h-7 items-center rounded border border-rose-300 bg-rose-100 px-2 font-semibold text-rose-800 text-xs transition-colors hover:bg-rose-200"
								>
									Spróbuj ponownie
								</button>
							</div>
						) : null}
					</>
				}
				footerContent={
					<>
						{formError ? (
							<div className="mb-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-rose-700 text-sm">
								<p className="font-medium">{formError}</p>
								{versionConflictUpdatedAt ? (
									<p className="mt-1 text-rose-700/90">
										Aktualna wersja rekordu: {versionConflictUpdatedAt}
									</p>
								) : null}
							</div>
						) : null}

						{versionConflictUpdatedAt ? (
							<div className="mb-2">
								<button
									type="button"
									onClick={() => void handleRefreshAfterConflict()}
									className="inline-flex h-8 items-center rounded-md border border-amber-300 bg-amber-50 px-3 font-semibold text-amber-800 text-xs transition-colors hover:bg-amber-100"
								>
									Odśwież dane
								</button>
							</div>
						) : null}
					</>
				}
				isSubmitDisabled={isSubmitting || isReadOnlyDueToLock || isSaveDisabledDueToLock}
				cancelLabel={isEditMode ? "Usuń zmiany" : "Anuluj"}
				submitLabel={
					isSubmitting
						? "Zapisywanie..."
						: isReadOnlyDueToLock
							? "Tylko podgląd"
							: isSaveDisabledDueToLock
								? "Brak blokady"
								: editingItem
									? "Zapisz"
									: "Dodaj"
				}
			>
				<div className="grid gap-3 sm:grid-cols-2">
								<div className="text-slate-700 text-sm">
									<SingleSelectPortalField
										label="Id inspekcji *"
										value={form.inspectionId}
										options={inspectionSelectOptions}
										placeholder={
											isInspectionOptionsLoading
												? "Ładowanie listy inspekcji..."
												: "Wybierz id inspekcji"
										}
										invalid={isRequiredInspectionMissing}
										errorMessage={
											isRequiredInspectionMissing ? "Pole wymagane." : null
										}
										onChange={(next) => {
											const selectedOption = inspectionOptions.find(
												(option) => String(option.id) === next,
											);
											setForm((prev) => ({
												...prev,
												inspectionId: next,
												nazwaPodmiotu: selectedOption?.nazwaPodmiotu ?? prev.nazwaPodmiotu,
											}));
										}}
										disabled={form.isInspectionMissing || isInspectionOptionsLoading}
									/>
									<label className="mt-2 inline-flex items-center gap-2 font-medium text-slate-700 text-xs">
										<input
											type="checkbox"
											checked={form.isInspectionMissing}
											onChange={(event) => {
												const checked = event.target.checked;
												setForm((prev) => ({
													...prev,
													isInspectionMissing: checked,
													inspectionId: checked ? "" : prev.inspectionId,
													nazwaPodmiotu: checked
														? prev.nazwaPodmiotu
														: prev.nazwaPodmiotu,
												}));
											}}
										/>
										Brak powiązania z kodem inspekcji
									</label>
									<input
										readOnly
										tabIndex={-1}
										aria-hidden="true"
										value={form.inspectionId}
										className="sr-only"
									/>
								</div>

								<div className="text-slate-700 text-sm">
									{form.isInspectionMissing ? (
										<SingleSelectPortalField
											label="Nazwa podmiotu *"
											value={form.nazwaPodmiotu}
											options={entityNameOptions}
											placeholder="Wybierz podmiot"
											invalid={isRequiredEntityNameMissing}
											errorMessage={
												isRequiredEntityNameMissing ? "Pole wymagane." : null
											}
											onChange={(next) =>
												setForm((prev) => ({
													...prev,
													nazwaPodmiotu: next,
												}))
											}
											disabled={isReadOnlyDueToLock}
										/>
									) : (
										<label className="text-slate-700 text-sm">
											<span className="mb-1 block">Nazwa podmiotu *</span>
											<input
												disabled
												value={form.nazwaPodmiotu}
												className="w-full cursor-not-allowed rounded-lg border border-slate-300 bg-slate-100 px-3 py-2 text-slate-700 text-sm outline-none"
											/>
										</label>
									)}
								</div>

								<label className="text-slate-700 text-sm">
									<span className="mb-1 block">Liczba zaleceń *</span>
									<input
										value={form.pozycja}
										onChange={(event) =>
											setForm((prev) => ({
												...prev,
												pozycja: event.target.value,
											}))
										}
										className={`w-full rounded-lg border px-3 py-2 text-sm outline-none transition-colors ${
											isRequiredPositionMissing ? "border-rose-300 focus:border-rose-400" : "border-slate-300 focus:border-blue-400"
										}`}
									/>
									{isRequiredPositionMissing ? (
										<span className="mt-1 block text-rose-700 text-xs">Pole wymagane.</span>
									) : null}
								</label>

								<DateFieldWithClear
									label="Data zaleceń"
									value={form.terminWykonaniaZalecen}
									onChange={(next) =>
										setForm((prev) => ({
											...prev,
											terminWykonaniaZalecen: next,
										}))
									}
								/>

								<div className="sm:col-span-2">
									<DateListEditor
										title="Termin wykonania zaleceń"
										addButtonLabel="Dodaj datę"
										noDatesLabel="Brak terminów wykonania zaleceń"
										noDatesMessage="Oznaczono brak terminów wykonania zaleceń."
										noDatesContext="terminów"
										values={form.dataZalecenList}
										setValues={setFormDataZalecenList}
										isNoDates={form.isDataZalecenBrak}
										setIsNoDates={setFormIsDataZalecenBrak}
										itemKeyPrefix="zalecenia"
									/>
								</div>

								<div className="sm:col-span-2">
									<DateListEditor
										title="Data akceptacji noty z weryfikacji wykonania zaleceń"
										addButtonLabel="Dodaj datę"
										noDatesLabel="Brak dat akceptacji noty"
										noDatesMessage="Oznaczono brak dat akceptacji noty."
										noDatesContext="akceptacji"
										values={form.dataAkceptacjiList}
										setValues={setFormDataAkceptacjiList}
										isNoDates={form.isDataAkceptacjiBrak}
										setIsNoDates={setFormIsDataAkceptacjiBrak}
										itemKeyPrefix="akceptacja-noty"
									/>
								</div>

								<div className="sm:col-span-2">
									<SingleSelectPortalField
										label="Status"
										value={form.status}
										options={statusOptionsForForm}
										placeholder="Wybierz status"
										onChange={(next) =>
											setForm((prev) => ({
												...prev,
												status: next,
											}))
										}
										disabled={isReadOnlyDueToLock}
									/>
								</div>

								<label className="text-slate-700 text-sm sm:col-span-2">
									<span className="mb-1 block">Komentarz</span>
									<textarea
										rows={2}
										value={form.komentarz}
										onChange={(event) =>
											setForm((prev) => ({
												...prev,
												komentarz: event.target.value,
											}))
										}
										className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none transition-colors focus:border-blue-400"
									/>
								</label>
				</div>
			</RegistryFormScaffold>

			<RecommendationsSuccessModal
				isOpen={isSuccessModalOpen}
				entityName={successEntityName}
				inspectionCode={successInspectionCode}
				mode={successMode}
				onClose={() => {
					setIsSuccessModalOpen(false);
					setSuccessEntityName("");
					setSuccessInspectionCode("");
					setSuccessMode("create");
				}}
			/>

			<DeleteSuccessModal
				isOpen={isDeleteSuccessModalOpen}
				heading="Zalecenie zostało usunięte"
				detailsMessage={
					deleteSuccessEntityName
						? `Dla podmiotu ${deleteSuccessEntityName}.`
						: "Rekord został usunięty z tabeli."
				}
				onClose={() => {
					setIsDeleteSuccessModalOpen(false);
					setDeleteSuccessEntityName("");
				}}
			/>

			{isDeleteConfirmModalOpen ? (
				<div className="fixed inset-0 z-60 flex items-center justify-center p-4">
					<button
						type="button"
						aria-label="Zamknij potwierdzenie usunięcia zalecenia"
						className="absolute inset-0 bg-slate-950/65"
						onClick={() => {
							if (isDeletingItem) {
								return;
							}

							setIsDeleteConfirmModalOpen(false);
						}}
					/>

					<div
						role="dialog"
						aria-modal="true"
						aria-label="Potwierdzenie usunięcia zalecenia"
						className="relative z-10 w-full max-w-lg rounded-2xl border border-slate-300 bg-white p-5 text-slate-900 shadow-[0_24px_56px_rgba(2,8,23,0.35)]"
					>
						<h3 className="font-semibold text-base text-slate-900">
							Usuń zalecenie
						</h3>
						<p className="mt-2 text-slate-700 text-sm">Czy usunąć zalecenie?</p>

						<div className="mt-5 flex items-center justify-end gap-2">
							<button
								type="button"
								onClick={() => setIsDeleteConfirmModalOpen(false)}
								disabled={isDeletingItem}
								className="inline-flex h-10 items-center rounded-lg border border-slate-300 bg-white px-4 font-semibold text-slate-700 text-sm transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
							>
								Anuluj
							</button>
							<button
								type="button"
								onClick={() => void handleDeleteItem()}
								disabled={isDeletingItem}
								className="inline-flex h-10 items-center rounded-lg border border-[#f2a3a3] bg-[#6f2a36] px-4 font-semibold text-[#ffe5e8] text-sm transition-colors hover:bg-[#833242] disabled:cursor-not-allowed disabled:opacity-60"
							>
								{isDeletingItem ? "Usuwanie..." : "Usuń"}
							</button>
						</div>
					</div>
				</div>
			) : null}
		</section>
	);
}
