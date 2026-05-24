"use client";

import { Pencil, Plus, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { AuthRole } from "@/app/_components/home-tabs/types";

import { fetchDictionaryEntries } from "@/features/dictionaries/api";
import type { DictionaryEntry } from "@/features/dictionaries/types";
import {
	type RawInspectionRow,
	normalizeInspectionRow,
} from "@/features/inspections/components/inspections-panel.utils";
import { fetchObligatingDecisions } from "@/features/obligating-decisions/api";
import { fetchRecommendations } from "@/features/recommendations/api";
import {
	createSanctionRequest,
	deleteSanctionRequest,
	fetchSanctionRequests,
	type SanctionRequestLockConflict,
	updateSanctionRequest,
} from "@/features/sanction-requests/api";
import { SanctionRequestsSuccessModal } from "./SanctionRequestsSuccessModal";
import type {
	SanctionRequestRead,
	SanctionRequestWrite,
} from "@/features/sanction-requests/types";
import { DeleteSuccessModal } from "@/shared/components/DeleteSuccessModal";
import { DateInputWithCalendar } from "@/shared/components/forms/DateInputWithCalendar";
import { RegistryFormScaffold } from "@/shared/components/forms/RegistryFormScaffold";
import { SingleSelectPortalField } from "@/shared/components/forms/SingleSelectPortalField";
import { ExportConfigModal } from "@/shared/components/export/ExportConfigModal";
import { TableAdvancedFilterModal } from "@/shared/components/table/TableAdvancedFilterModal";
import { TableColumnPickerModal } from "@/shared/components/table/TableColumnPickerModal";
import { TableHeaderWithFilters } from "@/shared/components/table/TableHeaderWithFilters";
import { TablePanelToolbar } from "@/shared/components/table/TablePanelToolbar";
import { TablePagination } from "@/shared/components/table/TablePagination";
import { TableSurface } from "@/shared/components/table/TableSurface";
import { useRecordLock } from "@/shared/hooks/useRecordLock";
import { useInactivityTimeout } from "@/shared/hooks/useInactivityTimeout";
import { useTableState } from "@/shared/hooks/useTableState";
import {
	addWorksheetWithStyles,
	createStyledExportWorkbook,
	saveWorkbookAsXlsx,
} from "@/shared/utils/excel-export";
import { getFloatingPanelAnchor } from "@/shared/utils/floating-panel";

const INACTIVITY_TIMEOUT_MS = 60_000; // 1 minuta (do testów)
const INACTIVITY_WARNING_MS = 30_000; // 30 s ostrzeżenie
const TABLE_PAGE_SIZE_OPTIONS = [20, 30, 50, 70, 100] as const;
const SANCTION_REQUESTS_COLUMN_WIDTHS_STORAGE_PREFIX =
	"triangle.ui.sanction-requests.column-widths";
const SANCTION_REQUESTS_NAME_VARIANTS_STORAGE_PREFIX =
	"triangle.ui.sanction-requests.name-variants";
const SANCTION_REQUESTS_MIN_COLUMN_WIDTH = 90;
const DASHBOARD_OPEN_INSPECTION_EVENT = "dashboard:open-inspection";
const DASHBOARD_OPEN_INSPECTION_CODE_KEY = "triangle.dashboard.openInspectionCode";

function openInspectionFromDashboard(inspectionCode: string) {
	const normalizedCode = inspectionCode.trim();
	if (!normalizedCode || typeof window === "undefined") {
		return;
	}

	window.sessionStorage.setItem(
		DASHBOARD_OPEN_INSPECTION_CODE_KEY,
		normalizedCode,
	);
	window.dispatchEvent(
		new CustomEvent(DASHBOARD_OPEN_INSPECTION_EVENT, {
			detail: { inspectionCode: normalizedCode },
		}),
	);
}

type SanctionRequestsPanelProps = {
	operatorLogin: string;
	authRole: AuthRole;
	isObserver?: boolean;
};

type SanctionRequestColumnKey =
	| "lp"
	| "requestId"
	| "inspectionId"
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

type SanctionRequestColumn = {
	key: SanctionRequestColumnKey;
	label: string;
};

type SanctionShortNameVariant = "full" | "short";

type SanctionShortNameColumnKey =
	| "nazwaPodmiotuObjetegoInspekcja"
	| "nazwaPodmiotuObjetegoSankcjaList"
	| "wniosekDo"
	| "sankcjaList"
	| "podstawaPrawnaSankcjiList"
	| "naruszeniaSkutkujaceSankcjaList"
	| "czyMamyInformacjeOWszczeciuPostepowania"
	| "rozstrzygniecie";

type SanctionShortNameVariantByColumn = Record<
	SanctionShortNameColumnKey,
	SanctionShortNameVariant
>;

const SANCTION_SHORT_NAME_COLUMN_KEYS: SanctionShortNameColumnKey[] = [
	"nazwaPodmiotuObjetegoInspekcja",
	"nazwaPodmiotuObjetegoSankcjaList",
	"wniosekDo",
	"sankcjaList",
	"podstawaPrawnaSankcjiList",
	"naruszeniaSkutkujaceSankcjaList",
	"czyMamyInformacjeOWszczeciuPostepowania",
	"rozstrzygniecie",
];

const SANCTION_SHORT_NAME_VARIANT_OPTIONS = [
	{ value: "full", label: "Nazwa pełna" },
	{ value: "short", label: "Nazwa skrócona" },
] as const;

const DEFAULT_SANCTION_SHORT_NAME_VARIANTS: SanctionShortNameVariantByColumn = {
	nazwaPodmiotuObjetegoInspekcja: "short",
	nazwaPodmiotuObjetegoSankcjaList: "short",
	wniosekDo: "full",
	sankcjaList: "full",
	podstawaPrawnaSankcjiList: "full",
	naruszeniaSkutkujaceSankcjaList: "full",
	czyMamyInformacjeOWszczeciuPostepowania: "full",
	rozstrzygniecie: "full",
};

function isSanctionShortNameColumnKey(
	columnKey: SanctionRequestColumnKey,
): columnKey is SanctionShortNameColumnKey {
	return SANCTION_SHORT_NAME_COLUMN_KEYS.includes(
		columnKey as SanctionShortNameColumnKey,
	);
}

const SANCTION_REQUEST_COLUMNS: SanctionRequestColumn[] = [
	{ key: "lp", label: "Lp." },
	{ key: "requestId", label: "Id wniosku" },
	{ key: "inspectionId", label: "Id inspekcji" },
	{
		key: "nazwaPodmiotuObjetegoInspekcja",
		label: "Nazwa podmiotu\nobjętego inspekcją",
	},
	{
		key: "nazwaPodmiotuObjetegoSankcjaList",
		label: "Nazwa podmiotu\nobjętego sankcją",
	},
	{ key: "dataWniosku", label: "Data wniosku" },
	{ key: "wniosekDo", label: "Wniosek do" },
	{ key: "sankcjaList", label: "Sankcja" },
	{
		key: "podstawaPrawnaSankcjiList",
		label: "Podstawa prawna\nsankcji",
	},
	{
		key: "naruszeniaSkutkujaceSankcjaList",
		label: "Naruszenia skutkujące\nsankcją",
	},
	{
		key: "czyMamyInformacjeOWszczeciuPostepowania",
		label: "Informacja o wszczęciu\npostępowania",
	},
	{ key: "rozstrzygniecie", label: "Rozstrzygnięcie" },
	{ key: "komentarz", label: "Komentarz" },
];

const SANCTION_REQUEST_COLUMN_TOOLTIPS: Partial<
	Record<SanctionRequestColumnKey, string>
> = {
	requestId: "Unikalne id wniosku sankcyjnego",
	inspectionId: "Unikalne id inspekcji",
};

const ALL_SANCTION_REQUEST_COLUMN_KEYS: SanctionRequestColumnKey[] =
	SANCTION_REQUEST_COLUMNS.map((column) => column.key);

const DEFAULT_SANCTION_REQUEST_COLUMN_WIDTHS: Partial<
	Record<SanctionRequestColumnKey, number>
> = {
	lp: 90,
	requestId: 170,
	inspectionId: 170,
	nazwaPodmiotuObjetegoInspekcja: 240,
	nazwaPodmiotuObjetegoSankcjaList: 250,
	dataWniosku: 170,
	wniosekDo: 190,
	sankcjaList: 220,
	podstawaPrawnaSankcjiList: 260,
	naruszeniaSkutkujaceSankcjaList: 280,
	czyMamyInformacjeOWszczeciuPostepowania: 260,
	rozstrzygniecie: 210,
	komentarz: 240,
};

type InspectionOption = {
	id: number;
	lp: number;
	inspectionCode: string;
	nazwaPodmiotu: string;
};

type SanctionRequestFormState = {
	inspectionId: string;
	isInspectionMissing: boolean;
	nazwaPodmiotuObjetegoInspekcja: string;
	nazwaPodmiotuObjetegoSankcjaList: string[];
	dataWniosku: string;
	wniosekDo: string;
	sankcjaList: string[];
	podstawaPrawnaSankcjiList: string[];
	naruszeniaSkutkujaceSankcjaList: string[];
	czyMamyInformacjeOWszczeciuPostepowania: string;
	rozstrzygniecie: string;
	komentarz: string;
};

const EMPTY_FORM: SanctionRequestFormState = {
	inspectionId: "",
	isInspectionMissing: false,
	nazwaPodmiotuObjetegoInspekcja: "",
	nazwaPodmiotuObjetegoSankcjaList: [],
	dataWniosku: "",
	wniosekDo: "",
	sankcjaList: [],
	podstawaPrawnaSankcjiList: [],
	naruszeniaSkutkujaceSankcjaList: [],
	czyMamyInformacjeOWszczeciuPostepowania: "",
	rozstrzygniecie: "",
	komentarz: "",
};

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

const INSPECTIONS_API_URL = "/api/structure/inspections";
const AVAILABLE_INSPECTIONS_API_URL =
	"/api/sanction-requests/available-inspections";
const AVAILABLE_INSPECTIONS_ALIAS_API_URL =
	"/api/risk-exposure/available-inspections";
const SANCTION_ENTITY_OPTIONS_API_URL =
	"/api/sanction-requests/entity-options";
const SANCTION_ENTITY_OPTIONS_ALIAS_API_URL =
	"/api/risk-exposure/entity-options";

type InspectionExportColumnKey =
	| "kodInspekcji"
	| "nazwaPodmiotu"
	| "typInspekcji"
	| "zakresInspekcji"
	| "szczegolyDotyczaceZakresu"
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

type RecommendationExportColumnKey =
	| "lp"
	| "kodZalecenia"
	| "inspectionLp"
	| "nazwaPodmiotu"
	| "pozycja"
	| "terminWykonaniaZalecen"
	| "dataZalecenList"
	| "dataAkceptacjiNotyWeryfikacjiList"
	| "status"
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

const INSPECTION_EXPORT_COLUMNS: ExportColumnDefinition<InspectionExportColumnKey>[] =
	[
		{ key: "kodInspekcji", label: "Id inspekcji" },
		{ key: "nazwaPodmiotu", label: "Nazwa podmiotu" },
		{ key: "typInspekcji", label: "Typ inspekcji" },
		{ key: "zakresInspekcji", label: "Zakres inspekcji według upoważnienia" },
		{
			key: "szczegolyDotyczaceZakresu",
			label: "Szczegóły dotyczące zakresu",
		},
		{ key: "aspektKonsumencki", label: "Aspekt konsumencki" },
		{ key: "poczatekInspekcji", label: "Początek inspekcji" },
		{ key: "koniecInspekcji", label: "Koniec inspekcji" },
		{ key: "osobaKierujaca", label: "Osoba kierująca kontrolą / wizytą" },
		{ key: "skladZespolu", label: "Skład zespołu inspekcyjnego" },
		{ key: "rynek", label: "Rynek" },
		{ key: "rodzajPodmiotu", label: "Rodzaj podmiotu" },
		{ key: "dataProtokolu", label: "Data protokołu / sprawozdania" },
		{ key: "dataDoreczeniaProtokolu", label: "Data doręczenia protokołu" },
		{
			key: "dataAkceptacjiSprawozdania",
			label: "Data akceptacji sprawozdania z wizyty",
		},
		{ key: "dataDoreczeniaPisma", label: "Data doręczenia pisma po wizycie" },
		{
			key: "dataPismaZastrzezenia",
			label: "Data pisma z zastrzeżeniami do protokołu / pisma po wizycie",
		},
		{
			key: "dataWyslaniaPismaZZastrzezeniami",
			label: "Data wysłania pisma z zastrzeżeniami",
		},
		{
			key: "dataWplywuPisma",
			label: "Data wpływu pisma z zastrzeżeniami do protokołu / pisma po wizycie",
		},
		{
			key: "dataPismaZOdpowiedzia",
			label: "Data pisma z odpowiedzią na zastrzeżenia",
		},
		{
			key: "dataWyslaniaPismaZOdpowiedzia",
			label: "Data wysłania pisma z odpowiedzią na zastrzeżenia",
		},
		{ key: "dataAkceptacjiNoty", label: "Data akceptacji noty" },
		{ key: "dataZalecen", label: "Data zaleceń" },
		{ key: "status", label: "Status" },
		{ key: "komentarz", label: "Komentarz" },
	];

const RECOMMENDATION_EXPORT_COLUMNS: ExportColumnDefinition<RecommendationExportColumnKey>[] =
	[
		{ key: "lp", label: "Lp." },
		{ key: "kodZalecenia", label: "Id zalecenia" },
		{ key: "inspectionLp", label: "Id inspekcji" },
		{ key: "nazwaPodmiotu", label: "Nazwa podmiotu" },
		{ key: "pozycja", label: "Liczba zaleceń" },
		{ key: "terminWykonaniaZalecen", label: "Data zaleceń" },
		{ key: "dataZalecenList", label: "Termin wykonania zaleceń" },
		{
			key: "dataAkceptacjiNotyWeryfikacjiList",
			label: "Data akceptacji noty z weryfikacji wykonania zaleceń",
		},
		{ key: "status", label: "Status" },
		{ key: "komentarz", label: "Komentarz" },
	];

const DECISION_EXPORT_COLUMNS: ExportColumnDefinition<DecisionExportColumnKey>[] =
	[
		{ key: "lp", label: "Lp." },
		{ key: "kodDecyzji", label: "Id decyzji" },
		{ key: "kodZalecenia", label: "Id zalecenia" },
		{ key: "inspectionLp", label: "Id inspekcji" },
		{ key: "nazwaPodmiotu", label: "Nazwa podmiotu" },
		{ key: "liczbaZalecen", label: "Liczba zaleceń" },
		{
			key: "dataWszczeciaPostepowaniaIInstancji",
			label: "Data wszczęcia postępowania administracyjnego I instancji",
		},
		{
			key: "osobyProwadzaceIInstancjeList",
			label: "Osoby prowadzące I instancję",
		},
		{ key: "dataDecyzjiIInstancji", label: "Data decyzji I instancji" },
		{
			key: "dataDoreczeniaDecyzjiIInstancji",
			label: "Data doręczenia decyzji I instancji",
		},
		{ key: "rozstrzygniecieI", label: "Rozstrzygnięcie I instancji" },
		{
			key: "dataWnioskuPonowneRozpatrzenie",
			label: "Data wniosku o ponowne rozpatrzenie sprawy",
		},
		{
			key: "dataWplywuWnioskuPonowneRozpatrzenie",
			label: "Data wpływu wniosku o ponowne rozpatrzenie sprawy",
		},
		{
			key: "osobyProwadzaceIIInstancjeList",
			label: "Osoby prowadzące II instancję",
		},
		{ key: "dataDecyzjiIIInstancji", label: "Data decyzji II instancji" },
		{
			key: "dataDoreczeniaDecyzjiIIInstancji",
			label: "Data doręczenia decyzji II instancji",
		},
		{ key: "rozstrzygniecieII", label: "Rozstrzygnięcie II instancji" },
		{ key: "komentarz", label: "Komentarz" },
	];

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

function normalizeStringList(values: string[]) {
	const normalized = values.map((value) => value.trim()).filter(Boolean);

	return Array.from(new Set(normalized)).sort((left, right) =>
		left.localeCompare(right, "pl", { sensitivity: "base" }),
	);
}

function shortenInsuranceEntityName(name: string) {
	const trimmed = name.trim();
	if (!trimmed) {
		return "";
	}

	const withoutRepeatedWords = trimmed.replace(
		/\b([\p{L}\p{N}.-]+)(?:\s+\1\b)+/giu,
		"$1",
	);

	return withoutRepeatedWords
		.replace(/Towarzystwo\s+Ubezpieczen\s+na\s+Zycie/gi, "TU na Zycie")
		.replace(/Towarzystwo\s+Ubezpieczeń\s+na\s+Życie/gi, "TU na Życie")
		.replace(/Towarzystwo\s+Ubezpieczen/gi, "TU")
		.replace(/Towarzystwo\s+Ubezpieczeń/gi, "TU")
		.replace(/Zaklad\s+Ubezpieczen/gi, "ZU")
		.replace(/Zakład\s+Ubezpieczeń/gi, "ZU")
		.replace(/Spolka\s+Akcyjna/gi, "SA")
		.replace(/Spółka\s+Akcyjna/gi, "SA")
		.replace(/Spolka\s+z\s+ograniczona\s+odpowiedzialnoscia/gi, "Sp. z o.o.")
		.replace(/Spółka\s+z\s+ograniczoną\s+odpowiedzialnością/gi, "Sp. z o.o.")
		.replace(/\s{2,}/g, " ")
		.trim();
}

function getCellValue(
	item: SanctionRequestRead,
	key: SanctionRequestColumnKey,
) {
	if (
		key === "nazwaPodmiotuObjetegoSankcjaList" ||
		key === "sankcjaList" ||
		key === "podstawaPrawnaSankcjiList" ||
		key === "naruszeniaSkutkujaceSankcjaList"
	) {
		return item[key].join("; ");
	}

	if (key === "inspectionId") {
		if (!item.inspectionId) {
			return "Brak";
		}
		return String(item.inspectionId);
	}

	if (key === "requestId") {
		return String(item.id);
	}

	const raw = item[key];
	if (raw === null || raw === undefined) {
		return "";
	}

	return String(raw);
}

function requestToForm(item: SanctionRequestRead): SanctionRequestFormState {
	return {
		inspectionId: item.inspectionId ? String(item.inspectionId) : "",
		isInspectionMissing: item.inspectionId === null,
		nazwaPodmiotuObjetegoInspekcja: item.nazwaPodmiotuObjetegoInspekcja ?? "",
		nazwaPodmiotuObjetegoSankcjaList: normalizeStringList(
			item.nazwaPodmiotuObjetegoSankcjaList,
		),
		dataWniosku: item.dataWniosku ?? "",
		wniosekDo: item.wniosekDo ?? "",
		sankcjaList: normalizeStringList(item.sankcjaList),
		podstawaPrawnaSankcjiList: normalizeStringList(
			item.podstawaPrawnaSankcjiList,
		),
		naruszeniaSkutkujaceSankcjaList: normalizeStringList(
			item.naruszeniaSkutkujaceSankcjaList,
		),
		czyMamyInformacjeOWszczeciuPostepowania:
			item.czyMamyInformacjeOWszczeciuPostepowania ?? "",
		rozstrzygniecie: item.rozstrzygniecie ?? "",
		komentarz: item.komentarz ?? "",
	};
}

function formToPayload(
	form: SanctionRequestFormState,
): SanctionRequestWrite | null {
	const inspectionId = Number(form.inspectionId);

	if (
		!form.isInspectionMissing &&
		(!Number.isFinite(inspectionId) || inspectionId <= 0)
	) {
		return null;
	}

	const manualEntityName = form.nazwaPodmiotuObjetegoInspekcja.trim();

	return {
		inspectionId: form.isInspectionMissing ? null : inspectionId,
		nazwaPodmiotuObjetegoInspekcja: form.isInspectionMissing
			? manualEntityName || null
			: null,
		nazwaPodmiotuObjetegoSankcjaList: normalizeStringList(
			form.nazwaPodmiotuObjetegoSankcjaList,
		),
		dataWniosku: form.dataWniosku || null,
		wniosekDo: form.wniosekDo.trim() || null,
		sankcjaList: normalizeStringList(form.sankcjaList),
		podstawaPrawnaSankcjiList: normalizeStringList(
			form.podstawaPrawnaSankcjiList,
		),
		naruszeniaSkutkujaceSankcjaList: normalizeStringList(
			form.naruszeniaSkutkujaceSankcjaList,
		),
		czyMamyInformacjeOWszczeciuPostepowania:
			form.czyMamyInformacjeOWszczeciuPostepowania.trim() || null,
		rozstrzygniecie: form.rozstrzygniecie.trim() || null,
		komentarz: form.komentarz.trim() || null,
	};
}

type MultiSelectFieldProps = {
	label: string;
	options: string[] | Array<{ value: string; label: string }>;
	values: string[];
	onChange: (next: string[]) => void;
	enableSearch?: boolean;
	searchPlaceholder?: string;
	disabled?: boolean;
	placeholder?: string;
	allowCustomValue?: boolean;
	onAddCustomValue?: (value: string) => void;
	customAddLabel?: string;
};

function MultiSelectField({
	label,
	options,
	values,
	onChange,
	enableSearch = false,
	searchPlaceholder = "Wyszukaj...",
	disabled = false,
	placeholder = "Wybierz",
	allowCustomValue = false,
	onAddCustomValue,
	customAddLabel = "Dodaj pozycję",
}: MultiSelectFieldProps) {
	const [isOpen, setIsOpen] = useState(false);
	const [customValueInput, setCustomValueInput] = useState("");
	const [searchQuery, setSearchQuery] = useState("");
	const triggerRef = useRef<HTMLButtonElement | null>(null);
	const popupRef = useRef<HTMLDivElement | null>(null);
	const [popupPosition, setPopupPosition] = useState<{
		top: number;
		left: number;
		width: number;
		maxHeight: number;
	} | null>(null);
	const baseOptions = options.map((option) =>
		typeof option === "string"
			? { value: option, label: option }
			: { value: option.value, label: option.label },
	);
	const normalizedOptions = values.reduce(
		(acc, selectedValue) => {
			if (acc.some((option) => option.value === selectedValue)) {
				return acc;
			}

			return [...acc, { value: selectedValue, label: selectedValue }];
		},
		baseOptions,
	);
	const labelByValue = new Map(
		normalizedOptions.map((option) => [option.value, option.label]),
	);
	const normalizedSearchQuery = searchQuery.trim().toLocaleLowerCase("pl-PL");
	const visibleOptions = normalizedSearchQuery
		? normalizedOptions.filter((option) =>
				option.label.toLocaleLowerCase("pl-PL").includes(normalizedSearchQuery),
		  )
		: normalizedOptions;
	const MAX_VISIBLE_OPTIONS = 6;
	const OPTION_ROW_HEIGHT_ESTIMATE = 42;
	const POPUP_VERTICAL_PADDING = 20;
	const POPUP_MIN_HEIGHT = 140;
	const POPUP_GAP = 8;
	const visibleOptionsCount = Math.min(
		MAX_VISIBLE_OPTIONS,
		Math.max(1, normalizedOptions.length),
	);
	const estimatedOptionsHeight =
		visibleOptionsCount * OPTION_ROW_HEIGHT_ESTIMATE + POPUP_VERTICAL_PADDING;

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
		const popupContentHeight = popupRef.current
			? popupRef.current.scrollHeight
			: estimatedOptionsHeight;
		const desiredHeight = Math.max(
			POPUP_MIN_HEIGHT,
			Math.min(popupContentHeight, estimatedOptionsHeight),
		);
		const spaceBelow = Math.max(0, availableBottom - rect.bottom - POPUP_GAP);
		const spaceAbove = Math.max(0, rect.top - availableTop - POPUP_GAP);
		const shouldOpenUp =
			spaceBelow < Math.min(desiredHeight, 180) && spaceAbove > spaceBelow;
		const maxHeight = Math.max(
			POPUP_MIN_HEIGHT,
			shouldOpenUp ? spaceAbove : spaceBelow,
		);
		const requestedHeight = Math.min(desiredHeight, maxHeight);
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
			maxHeight,
		});
	};

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
		const handleAnyScroll = (event: Event) => {
			const target = event.target as Node | null;
			if (target && popupRef.current?.contains(target)) {
				return;
			}
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
			if (popupRef.current) {
				resizeObserver.observe(popupRef.current);
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
	}, [estimatedOptionsHeight, isOpen, normalizedOptions.length]);

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

	const displayValue =
		values.length > 0
			? values
					.map((value) => labelByValue.get(value) ?? value)
					.join("; ")
			: placeholder;

	const toggleOption = (optionValue: string) => {
		if (disabled) {
			return;
		}

		if (values.includes(optionValue)) {
			onChange(values.filter((value) => value !== optionValue));
			return;
		}

		onChange([...values, optionValue]);
	};

	const handleAddCustomValue = () => {
		if (disabled) {
			return;
		}

		const normalized = customValueInput.trim();
		if (!normalized) {
			return;
		}

		onAddCustomValue?.(normalized);

		if (!values.includes(normalized)) {
			onChange([...values, normalized]);
		}

		setCustomValueInput("");
	};

	return (
		<label className="text-sm text-slate-700">
			<span className="mb-1 block">{label}</span>
			<div className="relative">
				<button
					ref={triggerRef}
					type="button"
					disabled={disabled}
					onClick={() => setIsOpen((prev) => !prev)}
					className="flex w-full items-start justify-between gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-left text-slate-900 text-sm outline-none transition-colors hover:bg-slate-50 focus:border-blue-400 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-600"
				>
					<span className="min-w-0 whitespace-normal break-words">{displayValue}</span>
					<X size={14} className="rotate-45 text-slate-500" />
				</button>

				{isOpen && popupPosition
					? createPortal(
							<div
								ref={popupRef}
								className="fixed z-[80] rounded-xl border border-slate-200 bg-white p-2 shadow-[0_14px_34px_rgba(15,23,42,0.14)]"
								style={{
									top: popupPosition.top,
									left: popupPosition.left,
									width: popupPosition.width,
									maxHeight: popupPosition.maxHeight,
									overflowY: "auto",
								}}
							>
						{allowCustomValue ? (
							<div className="mb-2 flex items-center gap-2 border-slate-200 border-b pb-2">
								<input
									type="text"
									value={customValueInput}
									disabled={disabled}
									onChange={(event) => setCustomValueInput(event.target.value)}
									onKeyDown={(event) => {
										if (event.key === "Enter") {
											event.preventDefault();
											handleAddCustomValue();
										}
									}}
									placeholder={customAddLabel}
									className="h-8 flex-1 rounded-md border border-slate-300 px-2 text-sm outline-none transition-colors focus:border-blue-400"
								/>
								<button
									type="button"
									disabled={disabled}
									onClick={handleAddCustomValue}
									className="inline-flex h-8 items-center rounded-md border border-[#6ea3f0] bg-[#2d4d7f] px-2.5 font-semibold text-slate-100 text-xs transition-colors hover:bg-[#375f99] disabled:cursor-not-allowed disabled:opacity-60"
								>
									Dodaj
								</button>
							</div>
						) : null}

						<div className="mb-2 border-slate-200 border-b pb-2 font-medium text-slate-600 text-xs">
							Wybierz jedną lub więcej pozycji
						</div>

						{enableSearch ? (
							<div className="mb-2">
								<input
									type="text"
									value={searchQuery}
									onChange={(event) => setSearchQuery(event.target.value)}
									placeholder={searchPlaceholder}
									className="h-8 w-full rounded-md border border-slate-300 px-2 text-sm outline-none transition-colors focus:border-blue-400"
								/>
							</div>
						) : null}

						<div className="subtle-vertical-scroll max-h-52 space-y-1 overflow-y-auto pr-1">
							{visibleOptions.length === 0 ? (
								<p className="px-2 py-1 text-slate-500 text-sm">
									Brak dostępnych opcji.
								</p>
							) : null}

							{visibleOptions.map((option) => {
								const isSelected = values.includes(option.value);
								return (
									<button
										key={`${option.value}-${option.label}`}
										type="button"
										disabled={disabled}
										onClick={() => toggleOption(option.value)}
										className={`flex w-full items-center gap-2 rounded-sm px-3 py-2.5 text-left text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
											isSelected
												? "bg-blue-100 font-medium text-blue-900"
												: "text-slate-900 hover:bg-blue-50 hover:text-blue-900"
										}`}
									>
										<input
											type="checkbox"
											checked={isSelected}
											disabled={disabled}
											readOnly
											className="h-4 w-4 accent-blue-600"
										/>
										<span className="min-w-0 whitespace-normal break-words">
											{option.label}
										</span>
									</button>
								);
							})}
						</div>
							</div>,
							document.body,
						)
					: null}
			</div>
			<span className="mt-1 block text-xs text-slate-500">
				Wybrano: {values.length}
			</span>
		</label>
	);
}

export function SanctionRequestsPanel({
	operatorLogin,
	authRole,
	isObserver,
}: SanctionRequestsPanelProps) {
	const [items, setItems] = useState<SanctionRequestRead[]>([]);
	const [total, setTotal] = useState(0);
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	const [selectedId, setSelectedId] = useState<number | null>(null);
	const [isFormOpen, setIsFormOpen] = useState(false);
	const [editingItem, setEditingItem] = useState<SanctionRequestRead | null>(
		null,
	);
	const [form, setForm] = useState<SanctionRequestFormState>(EMPTY_FORM);
	const [formError, setFormError] = useState<string | null>(null);
	const [showRequiredFieldErrors, setShowRequiredFieldErrors] = useState(false);
	const [versionConflictUpdatedAt, setVersionConflictUpdatedAt] = useState<
		string | null
	>(null);
	const [saveLockConflict, setSaveLockConflict] =
		useState<SanctionRequestLockConflict | null>(null);
	const [isSubmitting, setIsSubmitting] = useState(false);
	const [isSuccessModalOpen, setIsSuccessModalOpen] = useState(false);
	const [successEntityName, setSuccessEntityName] = useState("");
	const [successInspectionCode, setSuccessInspectionCode] = useState("");
	const [successMode, setSuccessMode] = useState<"create" | "edit">("create");
	const [isDeleteConfirmModalOpen, setIsDeleteConfirmModalOpen] =
		useState(false);
	const [isDeletingItem, setIsDeletingItem] = useState(false);
	const [isDeleteSuccessModalOpen, setIsDeleteSuccessModalOpen] =
		useState(false);
	const [deleteSuccessEntityName, setDeleteSuccessEntityName] = useState("");
	const [tablePageSize, setTablePageSize] = useState<number>(30);
	const [columnWidths, setColumnWidths] = useState<
		Partial<Record<SanctionRequestColumnKey, number>>
	>(DEFAULT_SANCTION_REQUEST_COLUMN_WIDTHS);
	const [sanctionShortNameVariants, setSanctionShortNameVariants] =
		useState<SanctionShortNameVariantByColumn>(
			DEFAULT_SANCTION_SHORT_NAME_VARIANTS,
		);
	const [draftSanctionShortNameVariants, setDraftSanctionShortNameVariants] =
		useState<SanctionShortNameVariantByColumn>(
			DEFAULT_SANCTION_SHORT_NAME_VARIANTS,
		);
	const [areNameVariantsHydrated, setAreNameVariantsHydrated] =
		useState(false);
	const [areColumnWidthsHydrated, setAreColumnWidthsHydrated] = useState(false);
	const canManageSanctionRequests = authRole !== "external_user" && !isObserver;
	const isDirector = authRole === "director";
	const normalizedOperatorLogin = operatorLogin.trim().toLowerCase();
	const columnWidthsStorageKey = `${SANCTION_REQUESTS_COLUMN_WIDTHS_STORAGE_PREFIX}.${normalizedOperatorLogin}`;
	const nameVariantsStorageKey = `${SANCTION_REQUESTS_NAME_VARIANTS_STORAGE_PREFIX}.${normalizedOperatorLogin}`;
	const [advancedFilterAnchor, setAdvancedFilterAnchor] = useState({
		top: 120,
		left: 120,
	});
	const [isExporting, setIsExporting] = useState(false);
	const [isExportConfigModalOpen, setIsExportConfigModalOpen] = useState(false);
	const [includeInspectionsInExport, setIncludeInspectionsInExport] =
		useState(false);
	const [includeRecommendationsInExport, setIncludeRecommendationsInExport] =
		useState(false);
	const [includeDecisionsInExport, setIncludeDecisionsInExport] =
		useState(false);
	const [activeExportColumnsTab, setActiveExportColumnsTab] = useState<
		"inspections" | "recommendations" | "decisions"
	>("inspections");
	const [selectedInspectionExportColumns, setSelectedInspectionExportColumns] =
		useState<InspectionExportColumnKey[]>(
			INSPECTION_EXPORT_COLUMNS.map((column) => column.key),
		);
	const [
		selectedRecommendationExportColumns,
		setSelectedRecommendationExportColumns,
	] = useState<RecommendationExportColumnKey[]>(
		RECOMMENDATION_EXPORT_COLUMNS.map((column) => column.key),
	);
	const [selectedDecisionExportColumns, setSelectedDecisionExportColumns] =
		useState<DecisionExportColumnKey[]>(
			DECISION_EXPORT_COLUMNS.map((column) => column.key),
		);

	const [inspectionOptions, setInspectionOptions] = useState<
		InspectionOption[]
	>([]);
	const [isInspectionOptionsLoading, setIsInspectionOptionsLoading] =
		useState(false);
	const [nazwaPodmiotuSankcjaOptions, setNazwaPodmiotuSankcjaOptions] =
		useState<Array<{ value: string; label: string }>>([]);
	const [nazwaPodmiotuInspekcjaOptions, setNazwaPodmiotuInspekcjaOptions] =
		useState<string[]>([]);
	const [wniosekDoOptions, setWniosekDoOptions] = useState<
		Array<{ value: string; label: string }>
	>([]);
	const [sankcjaOptions, setSankcjaOptions] = useState<
		Array<{ value: string; label: string }>
	>([]);
	const [podstawaPrawnaOptions, setPodstawaPrawnaOptions] = useState<
		Array<{ value: string; label: string }>
	>([]);
	const [naruszeniaOptions, setNaruszeniaOptions] = useState<
		Array<{ value: string; label: string }>
	>([]);
	const [informacjaOptions, setInformacjaOptions] = useState<
		Array<{ value: string; label: string }>
	>([]);
	const [rozstrzygniecieOptions, setRozstrzygniecieOptions] = useState<
		Array<{ value: string; label: string }>
	>([]);

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
					option.nazwaPodmiotu
						? ` - ${shortenInsuranceEntityName(option.nazwaPodmiotu)}`
						: ""
				}`,
			})),
		[inspectionOptions],
	);

	const isEditMode = Boolean(editingItem);
	const editRecordLock = useRecordLock({
		enabled: isFormOpen && isEditMode,
		module: "sanction-requests",
		recordId: editingItem?.id ?? null,
		operatorLogin,
		heartbeatIntervalMs: 20_000,
	});
	const shouldShowLockedByOtherUser =
		Boolean(saveLockConflict) || editRecordLock.isBlocked;
	const isReadOnlyDueToLock = isEditMode && shouldShowLockedByOtherUser;
	const lockOwnerDisplayName =
		saveLockConflict?.ownerDisplayName ||
		editRecordLock.owner?.displayName ||
		"";
	const lockOwnerLogin =
		saveLockConflict?.ownerLogin || editRecordLock.owner?.login || "";
	const lockOwnerLabel =
		lockOwnerDisplayName || lockOwnerLogin
			? `${lockOwnerDisplayName || "Nieznany użytkownik"}${
					lockOwnerLogin ? ` (${lockOwnerLogin})` : ""
				}`
			: "inny użytkownik";
	const lockAcquiredAt =
		saveLockConflict?.acquiredAt ||
		editRecordLock.lockDetails?.acquiredAt ||
		null;

	const isSaveDisabledDueToLock =
		isEditMode &&
		(editRecordLock.isAcquireFailed ||
			editRecordLock.isConnectionLost ||
			editRecordLock.isExpired);
	const closeModalRef = useRef<() => void>(() => {});
	const inactivityTimeout = useInactivityTimeout({
		enabled: isFormOpen,
		inactivityMs: INACTIVITY_TIMEOUT_MS,
		warningMs: INACTIVITY_WARNING_MS,
		onTimeout: () => closeModalRef.current(),
	});

	const inspectionCodeById = useMemo(
		() =>
			new Map(
				inspectionOptions.map((option) => [option.id, option.inspectionCode]),
			),
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
		item: SanctionRequestRead,
		key: SanctionRequestColumnKey,
	) => {
		if (key === "requestId") {
			const sanctionCode = String(item.kodSankcji ?? "").trim();
			if (sanctionCode) {
				return sanctionCode;
			}

			if (Number.isFinite(item.id) && item.id > 0) {
				return String(item.id);
			}

			return "";
		}

		if (
			key === "nazwaPodmiotuObjetegoSankcjaList" ||
			key === "sankcjaList" ||
			key === "podstawaPrawnaSankcjiList" ||
			key === "naruszeniaSkutkujaceSankcjaList"
		) {
			return item[key].join("; ");
		}

		if (key === "inspectionId") {
			return (
				resolveInspectionCode({
					inspectionKod: item.inspectionKod,
					kodInspekcji: item.kodInspekcji,
					inspectionLp: item.inspectionLp,
					inspectionId: item.inspectionId,
				}) || "-"
			);
		}

		const raw = item[key];
		if (raw === null || raw === undefined) {
			return "";
		}

		return String(raw);
	};

	const resolvedNazwaPodmiotuSankcjaSelectOptions = useMemo(
		() => {
			const uniqueByValue = new Map<string, { value: string; label: string }>();

			for (const option of nazwaPodmiotuSankcjaOptions) {
				const value = option.value.trim();
				if (!value) {
					continue;
				}

				const label = option.label.trim() || value;
				if (!uniqueByValue.has(value)) {
					uniqueByValue.set(value, { value, label });
				}
			}

			for (const selectedValue of form.nazwaPodmiotuObjetegoSankcjaList) {
				const value = selectedValue.trim();
				if (!value || uniqueByValue.has(value)) {
					continue;
				}

				uniqueByValue.set(value, {
					value,
					label: shortenInsuranceEntityName(value) || value,
				});
			}

			return Array.from(uniqueByValue.values()).sort((left, right) =>
				left.label.localeCompare(right.label, "pl", {
					sensitivity: "base",
				}),
			);
		},
		[form.nazwaPodmiotuObjetegoSankcjaList, nazwaPodmiotuSankcjaOptions],
	);

	const sanctionRequestRowsForDisplay = useMemo(
		() =>
			items.map((item) => {
				const withScalarFallback = (
					fullValue: string | null,
					shortValue: string | null,
				) => {
					const normalizedShortValue = String(shortValue ?? "").trim();
					if (normalizedShortValue) {
						return normalizedShortValue;
					}

					return fullValue;
				};

				const withListFallback = (fullValue: string[], shortValue: string[]) =>
					Array.isArray(shortValue) && shortValue.length > 0
						? shortValue
						: fullValue;

				return {
					...item,
					nazwaPodmiotuObjetegoInspekcja:
						sanctionShortNameVariants.nazwaPodmiotuObjetegoInspekcja === "short"
							? withScalarFallback(
									item.nazwaPodmiotuObjetegoInspekcja,
									item.nazwaPodmiotuObjetegoInspekcjaSkrocona,
								)
							: item.nazwaPodmiotuObjetegoInspekcja,
					nazwaPodmiotuObjetegoSankcjaList:
						sanctionShortNameVariants.nazwaPodmiotuObjetegoSankcjaList === "short"
							? withListFallback(
									item.nazwaPodmiotuObjetegoSankcjaList,
									item.nazwaPodmiotuObjetegoSankcjaListSkrocona,
								)
							: item.nazwaPodmiotuObjetegoSankcjaList,
					wniosekDo:
						sanctionShortNameVariants.wniosekDo === "short"
							? withScalarFallback(item.wniosekDo, item.wniosekDoSkrocona)
							: item.wniosekDo,
					sankcjaList:
						sanctionShortNameVariants.sankcjaList === "short"
							? withListFallback(item.sankcjaList, item.sankcjaListSkrocona)
							: item.sankcjaList,
					podstawaPrawnaSankcjiList:
						sanctionShortNameVariants.podstawaPrawnaSankcjiList === "short"
							? withListFallback(
									item.podstawaPrawnaSankcjiList,
									item.podstawaPrawnaSankcjiListSkrocona,
								)
							: item.podstawaPrawnaSankcjiList,
					naruszeniaSkutkujaceSankcjaList:
						sanctionShortNameVariants.naruszeniaSkutkujaceSankcjaList === "short"
							? withListFallback(
									item.naruszeniaSkutkujaceSankcjaList,
									item.naruszeniaSkutkujaceSankcjaListSkrocona,
								)
							: item.naruszeniaSkutkujaceSankcjaList,
					czyMamyInformacjeOWszczeciuPostepowania:
						sanctionShortNameVariants.czyMamyInformacjeOWszczeciuPostepowania ===
						"short"
							? withScalarFallback(
									item.czyMamyInformacjeOWszczeciuPostepowania,
									item.czyMamyInformacjeOWszczeciuPostepowaniaSkrocona,
								)
							: item.czyMamyInformacjeOWszczeciuPostepowania,
					rozstrzygniecie:
						sanctionShortNameVariants.rozstrzygniecie === "short"
							? withScalarFallback(
									item.rozstrzygniecie,
									item.rozstrzygniecieSkrocona,
								)
							: item.rozstrzygniecie,
				};
			}),
		[items, sanctionShortNameVariants],
	);

	const {
		advancedFilterColumnKey,
		advancedFilterSearch,
		advancedFilters,
		canClearFilters,
		clearAdvancedFilterForSelectedColumn,
		clearFilters,
		columnFilters,
		draftHiddenColumns,
		draftVisibleColumns: draftVisibleSanctionRequestColumns,
		filteredAndSortedRows: filteredAndSortedItems,
		paginatedRows: paginatedSanctionRequestItems,
		currentPage,
		totalPages,
		pageSize,
		paginationItems,
		handlePageChange,
		handleApplyViewChanges,
		handleDraftColumnVisibilityChange,
		handleDraftDeselectAllColumns,
		handleDraftSelectAllColumns,
		handleFilterChange,
		handleOpenViewModal,
		handleSortByColumn,
		isAdvancedFilterModalOpen,
		isColumnPickerOpen,
		selectedAdvancedFilterDateRange,
		selectedAdvancedFilterValues,
		selectAllVisibleAdvancedFilterValues,
		setAdvancedFilterColumnKey,
		setAdvancedFilterDateRange,
		setAdvancedFilterSearch,
		setIsAdvancedFilterModalOpen,
		setIsColumnPickerOpen,
		sortColumnKey,
		sortDirection,
		toggleAdvancedFilterValue,
		visibleAdvancedFilterValues,
		visibleColumns: visibleSanctionRequestColumns,
	} = useTableState<SanctionRequestRead, SanctionRequestColumnKey>({
		rows: sanctionRequestRowsForDisplay,
		allColumnKeys: ALL_SANCTION_REQUEST_COLUMN_KEYS,
		initialAdvancedFilterColumnKey: "nazwaPodmiotuObjetegoInspekcja",
		getCellValue,
		pageSize: tablePageSize,
		sortComparators: {
			lp: (left, right) =>
				(Number(getCellValue(left, "lp")) || 0) -
				(Number(getCellValue(right, "lp")) || 0),
		},
	});

	const columnDisplayModeOptionsByKey = useMemo(
		() =>
			Object.fromEntries(
				SANCTION_SHORT_NAME_COLUMN_KEYS.map((columnKey) => [
					columnKey,
					[...SANCTION_SHORT_NAME_VARIANT_OPTIONS],
				]),
			) as Partial<
				Record<
					SanctionRequestColumnKey,
					Array<{ value: string; label: string }>
				>
			>,
		[],
	);

	const draftColumnDisplayModeValuesByKey = useMemo(
		() =>
			Object.fromEntries(
				SANCTION_SHORT_NAME_COLUMN_KEYS.map((columnKey) => [
					columnKey,
					draftSanctionShortNameVariants[columnKey],
				]),
			) as Partial<Record<SanctionRequestColumnKey, string>>,
		[draftSanctionShortNameVariants],
	);

	const handleOpenSanctionViewModal = () => {
		setDraftSanctionShortNameVariants(sanctionShortNameVariants);
		handleOpenViewModal();
	};

	const handleApplySanctionViewChanges = () => {
		setSanctionShortNameVariants(draftSanctionShortNameVariants);
		handleApplyViewChanges();
	};

	const handleResetSanctionViewSelection = () => {
		handleDraftSelectAllColumns();
		setDraftSanctionShortNameVariants(DEFAULT_SANCTION_SHORT_NAME_VARIANTS);
	};

	useEffect(() => {
		if (typeof window === "undefined") {
			return;
		}

		const raw = window.sessionStorage.getItem(nameVariantsStorageKey);
		if (!raw) {
			setAreNameVariantsHydrated(true);
			return;
		}

		try {
			const parsed = JSON.parse(raw) as Partial<
				Record<SanctionRequestColumnKey, unknown>
			>;
			const next: SanctionShortNameVariantByColumn = {
				...DEFAULT_SANCTION_SHORT_NAME_VARIANTS,
			};

			for (const columnKey of SANCTION_SHORT_NAME_COLUMN_KEYS) {
				const value = parsed[columnKey];
				if (value === "full" || value === "short") {
					next[columnKey] = value;
				}
			}

			setSanctionShortNameVariants(next);
		} catch {
			// ignore invalid persisted data
		}

		setAreNameVariantsHydrated(true);
	}, [nameVariantsStorageKey]);

	useEffect(() => {
		if (typeof window === "undefined") {
			return;
		}

		if (!areNameVariantsHydrated) {
			return;
		}

		window.sessionStorage.setItem(
			nameVariantsStorageKey,
			JSON.stringify(sanctionShortNameVariants),
		);
	}, [
		areNameVariantsHydrated,
		nameVariantsStorageKey,
		sanctionShortNameVariants,
	]);

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

	const visibleSanctionRequestColumnDefinitions = useMemo(
		() =>
			SANCTION_REQUEST_COLUMNS.filter((column) =>
				visibleSanctionRequestColumns.includes(column.key),
			),
		[visibleSanctionRequestColumns],
	);

	useEffect(() => {
		if (typeof window === "undefined") {
			return;
		}

		try {
			const raw = window.sessionStorage.getItem(columnWidthsStorageKey);
			if (!raw) {
				setColumnWidths(DEFAULT_SANCTION_REQUEST_COLUMN_WIDTHS);
				setAreColumnWidthsHydrated(true);
				return;
			}

			const parsed = JSON.parse(raw) as Partial<
				Record<SanctionRequestColumnKey, unknown>
			>;
			const sanitized: Partial<Record<SanctionRequestColumnKey, number>> = {
				...DEFAULT_SANCTION_REQUEST_COLUMN_WIDTHS,
			};

			for (const key of ALL_SANCTION_REQUEST_COLUMN_KEYS) {
				const value = parsed[key];
				if (typeof value !== "number" || !Number.isFinite(value)) {
					continue;
				}

				sanitized[key] = Math.max(
					SANCTION_REQUESTS_MIN_COLUMN_WIDTH,
					Math.round(value),
				);
			}

			setColumnWidths(sanitized);
		} catch {
			setColumnWidths(DEFAULT_SANCTION_REQUEST_COLUMN_WIDTHS);
		} finally {
			setAreColumnWidthsHydrated(true);
		}
	}, [columnWidthsStorageKey]);

	const hasCustomColumnWidths = useMemo(() => {
		const keys = new Set<SanctionRequestColumnKey>([
			...ALL_SANCTION_REQUEST_COLUMN_KEYS,
			...(Object.keys(columnWidths) as SanctionRequestColumnKey[]),
		]);

		for (const columnKey of keys) {
			const currentWidth = columnWidths[columnKey];
			const defaultWidth = DEFAULT_SANCTION_REQUEST_COLUMN_WIDTHS[columnKey];
			if (currentWidth !== defaultWidth) {
				return true;
			}
		}

		return false;
	}, [columnWidths]);

	useEffect(() => {
		if (typeof window === "undefined" || !areColumnWidthsHydrated) {
			return;
		}

		if (!hasCustomColumnWidths) {
			window.sessionStorage.removeItem(columnWidthsStorageKey);
			return;
		}

		window.sessionStorage.setItem(
			columnWidthsStorageKey,
			JSON.stringify(columnWidths),
		);
	}, [
		areColumnWidthsHydrated,
		columnWidths,
		columnWidthsStorageKey,
		hasCustomColumnWidths,
	]);

	const handleResizeColumn = (
		columnKey: SanctionRequestColumnKey,
		width: number,
	) => {
		if (!Number.isFinite(width)) {
			return;
		}

		setColumnWidths((prev) => ({
			...prev,
			[columnKey]: Math.max(SANCTION_REQUESTS_MIN_COLUMN_WIDTH, Math.round(width)),
		}));
	};

	const handleResetColumnWidths = () => {
		setColumnWidths(DEFAULT_SANCTION_REQUEST_COLUMN_WIDTHS);
	};

	const loadItems = async () => {
		setError(null);
		setIsLoading(true);

		const result = await fetchSanctionRequests(operatorLogin, {
			sortBy: "lp",
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
			let response = await fetch(AVAILABLE_INSPECTIONS_API_URL, {
				method: "GET",
				headers: {
					"Content-Type": "application/json",
					"X-Operator-Login": operatorLogin,
				},
				cache: "no-store",
			});

			if (response.status === 404) {
				response = await fetch(AVAILABLE_INSPECTIONS_ALIAS_API_URL, {
					method: "GET",
					headers: {
						"Content-Type": "application/json",
						"X-Operator-Login": operatorLogin,
					},
					cache: "no-store",
				});
			}

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

	const loadDictionaryOptions = async (
		kodTypu: string,
		setter: (value: string[]) => void,
	) => {
		try {
			const result = await fetchDictionaryEntries(kodTypu);
			if (!result.ok) {
				setter([]);
				return;
			}
			setter(mapDictionaryEntriesToOptions(result.data));
		} catch {
			setter([]);
		}
	};

	const loadDictionarySelectOptions = async (
		kodTypu: string,
		setter: (value: Array<{ value: string; label: string }>) => void,
	) => {
		try {
			const result = await fetchDictionaryEntries(kodTypu);
			if (!result.ok) {
				setter([]);
				return;
			}

			const mapped = result.data
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
				.map((entry) => {
					const value = entry.nazwaPozycji.trim();
					const shortLabel = (entry.skrotPozycji ?? "").trim();
					return {
						value,
						label: shortLabel || value,
					};
				})
				.filter((option) => option.value.length > 0);

			const uniqueByValue = new Map<string, { value: string; label: string }>();
			for (const option of mapped) {
				if (!uniqueByValue.has(option.value)) {
					uniqueByValue.set(option.value, option);
				}
			}

			setter(Array.from(uniqueByValue.values()));
		} catch {
			setter([]);
		}
	};

	const loadSanctionEntityOptions = async () => {
		const fallbackToDictionary = async () => {
			try {
				const result = await fetchDictionaryEntries("nazwy_podmiotow_sankcje");
				if (!result.ok) {
					setNazwaPodmiotuSankcjaOptions([]);
					return;
				}

				const mapped = mapDictionaryEntriesToOptions(result.data).map((value) => ({
					value,
					label: shortenInsuranceEntityName(value) || value,
				}));

				setNazwaPodmiotuSankcjaOptions(mapped);
			} catch {
				setNazwaPodmiotuSankcjaOptions([]);
			}
		};

		try {
			const searchParams = new URLSearchParams({
				includeHistorical: "true",
				limit: "1000",
				offset: "0",
			});

			let response = await fetch(
				`${SANCTION_ENTITY_OPTIONS_API_URL}?${searchParams.toString()}`,
				{
					method: "GET",
					headers: {
						"Content-Type": "application/json",
						"X-Operator-Login": operatorLogin,
					},
					cache: "no-store",
				},
			);

			if (response.status === 404) {
				response = await fetch(
					`${SANCTION_ENTITY_OPTIONS_ALIAS_API_URL}?${searchParams.toString()}`,
					{
						method: "GET",
						headers: {
							"Content-Type": "application/json",
							"X-Operator-Login": operatorLogin,
						},
						cache: "no-store",
					},
				);
			}

			if (!response.ok) {
				await fallbackToDictionary();
				return;
			}

			const payload = (await response.json()) as
				| Array<{
						value?: unknown;
						label?: unknown;
						source?: unknown;
						active?: unknown;
				  }>
				| {
						items?: Array<{
							value?: unknown;
							label?: unknown;
							source?: unknown;
							active?: unknown;
						}>;
				  };

			const rawItems = Array.isArray(payload) ? payload : (payload.items ?? []);
			const uniqueByValue = new Map<string, { value: string; label: string }>();

			for (const item of rawItems) {
				const value = String(item.value ?? "").trim();
				if (!value) {
					continue;
				}

				const isActive =
					typeof item.active === "boolean" ? item.active : true;
				if (!isActive) {
					continue;
				}

				const label = String(item.label ?? "").trim() || value;
				if (!uniqueByValue.has(value)) {
					uniqueByValue.set(value, { value, label });
				}
			}

			setNazwaPodmiotuSankcjaOptions(
				Array.from(uniqueByValue.values()).sort((left, right) =>
					left.label.localeCompare(right.label, "pl", {
						sensitivity: "base",
					}),
				),
			);
		} catch {
			await fallbackToDictionary();
		}
	};

	useEffect(() => {
		void loadItems();
		void loadDictionaryOptions(
			"nazwy_podmiotow",
			setNazwaPodmiotuInspekcjaOptions,
		);
		void loadSanctionEntityOptions();
		void loadDictionarySelectOptions("department", setWniosekDoOptions);
		void loadDictionarySelectOptions("sankcja", setSankcjaOptions);
		void loadDictionarySelectOptions(
			"podstawa_prawna_sankcji",
			setPodstawaPrawnaOptions,
		);
		void loadDictionarySelectOptions(
			"naruszenia_skutkujace_sankcja",
			setNaruszeniaOptions,
		);
		void loadDictionarySelectOptions(
			"informacja_o_wszczeciu_postepowania_sankcyjnego",
			setInformacjaOptions,
		);
		void loadDictionarySelectOptions(
			"rozstrzygniecie_wniosku_sankcyjnego_i",
			setRozstrzygniecieOptions,
		);
		void loadInspectionOptions();
	}, []);

	useEffect(() => {
		if (form.isInspectionMissing) {
			return;
		}

		if (!selectedInspectionOption) {
			setForm((prev) => ({
				...prev,
				nazwaPodmiotuObjetegoInspekcja: "",
			}));
			return;
		}

		setForm((prev) => ({
			...prev,
			nazwaPodmiotuObjetegoInspekcja: shortenInsuranceEntityName(
				selectedInspectionOption.nazwaPodmiotu,
			),
		}));
	}, [form.isInspectionMissing, selectedInspectionOption]);

	const openAdvancedFilterForColumn = (
		columnKey: SanctionRequestColumnKey,
		triggerElement: HTMLElement,
	) => {
		setAdvancedFilterAnchor(getFloatingPanelAnchor(triggerElement));
		setAdvancedFilterColumnKey(columnKey);
		setAdvancedFilterSearch("");
		setIsAdvancedFilterModalOpen(true);
	};

	const handleExportCurrentView = async (
		inspectionColumnKeys: InspectionExportColumnKey[],
		recommendationColumnKeys: RecommendationExportColumnKey[],
		decisionColumnKeys: DecisionExportColumnKey[],
		includeInspections: boolean,
		includeRecommendations: boolean,
		includeDecisions: boolean,
	) => {
		if (
			isExporting ||
			filteredAndSortedItems.length === 0 ||
			visibleSanctionRequestColumnDefinitions.length === 0
		) {
			return;
		}

		setIsExporting(true);
		setError(null);

		try {
			const workbook = await createStyledExportWorkbook(
				"Ewidencja wnioskow sankcyjnych",
			);

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
							.filter(
								(entry): entry is readonly [number, number] => entry !== null,
							),
					);
				} catch {
					return new Map<number, number>();
				}
			};

			const [
				inspectionsResponse,
				recommendationsResult,
				decisionsResult,
				recommendationsLpById,
			] = await Promise.all([
				fetch(INSPECTIONS_API_URL, {
					method: "GET",
					headers: {
						"Content-Type": "application/json",
						"X-Operator-Login": operatorLogin,
					},
					cache: "no-store",
				}),
				fetchRecommendations(operatorLogin, {
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

			const relatedRecommendationsSource = recommendationsResult.ok
				? recommendationsResult.data.items
				: [];
			const relatedDecisionsSource = decisionsResult.ok
				? decisionsResult.data.items
				: [];
			const relatedRecommendations = relatedRecommendationsSource.filter(
				(item) =>
					typeof item.inspectionId === "number" &&
					linkedInspectionIds.has(item.inspectionId),
			);

			const linkedRecommendationCodes = new Set(
				relatedRecommendations
					.map((item) =>
						String(item.kodZalecenia ?? "")
							.trim()
							.toUpperCase(),
					)
					.filter((value) => value.length > 0),
			);

			const relatedDecisions = relatedDecisionsSource.filter((item) => {
				const recommendationCode = String(item.recommendationKodZalecenia ?? "")
					.trim()
					.toUpperCase();

				return (
					recommendationCode.length > 0 &&
					linkedRecommendationCodes.has(recommendationCode)
				);
			});

			const relatedInspectionsForExport = mappedInspections.filter((row) =>
				linkedInspectionIds.has(Number(row.id)),
			);

			const inspectionCodeByIdForExport = new Map(
				relatedInspectionsForExport.map((row) => [
					Number(row.id),
					row.kodInspekcji,
				]),
			);

			const sanctionHeaders = visibleSanctionRequestColumnDefinitions.map(
				(column) => column.label,
			);
			const sanctionRows = filteredAndSortedItems.map((item) =>
				visibleSanctionRequestColumnDefinitions.map((column) =>
					getCellValue(item, column.key),
				),
			);

			addWorksheetWithStyles(
				workbook,
				"Wnioski sankcyjne",
				sanctionHeaders,
				sanctionRows,
			);

			if (includeInspections && inspectionColumnKeys.length > 0) {
				const inspectionHeaders = inspectionColumnKeys.map(
					(key) =>
						INSPECTION_EXPORT_COLUMNS.find((column) => column.key === key)
							?.label ?? key,
				);
				const inspectionRowsForExport = relatedInspectionsForExport.map((row) =>
					inspectionColumnKeys.map((key) => String(row[key] ?? "")),
				);
				addWorksheetWithStyles(
					workbook,
					"Inspekcje",
					inspectionHeaders,
					inspectionRowsForExport,
				);
			}

			if (includeRecommendations && recommendationColumnKeys.length > 0) {
				const recommendationHeaders = recommendationColumnKeys.map(
					(key) =>
						RECOMMENDATION_EXPORT_COLUMNS.find((column) => column.key === key)
							?.label ?? key,
				);
				const recommendationRows = relatedRecommendations.map((item) => {
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
										recommendationsLpById.get(inspectionId) ??
										"",
								)
							: "");

					return recommendationColumnKeys.map((key) => {
						switch (key) {
							case "lp":
								return String(item.lp);
							case "kodZalecenia":
								return String(item.kodZalecenia ?? "").trim();
							case "inspectionLp":
								return inspectionCode;
							case "nazwaPodmiotu":
								return item.nazwaPodmiotu;
							case "pozycja":
								return String(item.pozycja);
							case "terminWykonaniaZalecen":
								return item.terminWykonaniaZalecen ?? "";
							case "dataZalecenList":
								return item.dataZalecenList.join(", ");
							case "dataAkceptacjiNotyWeryfikacjiList":
								return item.dataAkceptacjiNotyWeryfikacjiList.join(", ");
							case "status":
								return item.status ?? "";
							case "komentarz":
								return item.komentarz ?? "";
						}
					});
				});
				addWorksheetWithStyles(
					workbook,
					"Zalecenia",
					recommendationHeaders,
					recommendationRows,
				);
			}

			if (includeDecisions && decisionColumnKeys.length > 0) {
				const decisionHeaders = decisionColumnKeys.map(
					(key) =>
						DECISION_EXPORT_COLUMNS.find((column) => column.key === key)
							?.label ?? key,
				);
				const decisionRows = relatedDecisions.map((item, index) =>
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
							case "inspectionLp": {
								const relatedRecommendation = relatedRecommendations.find(
									(recommendation) =>
										String(recommendation.kodZalecenia ?? "")
											.trim()
											.toUpperCase() === recommendationCode,
								);
								if (!relatedRecommendation) {
									return "";
								}

								const inspectionId = relatedRecommendation.inspectionId ?? null;
								return (
									resolveInspectionCode({
										inspectionKod: relatedRecommendation.inspectionKod,
										kodInspekcji: relatedRecommendation.kodInspekcji,
										inspectionLp: relatedRecommendation.inspectionLp,
										inspectionId,
									}) ||
									(typeof inspectionId === "number"
										? String(
												inspectionCodeByIdForExport.get(inspectionId) ??
													recommendationsLpById.get(inspectionId) ??
													"",
											)
										: "")
								);
							}
							case "nazwaPodmiotu":
								return item.nazwaPodmiotu ?? "";
							case "liczbaZalecen":
								return item.liczbaZalecen === null
									? ""
									: String(item.liczbaZalecen);
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
					decisionRows,
				);
			}

			const fileName = "wnioski-sankcyjne-inspekcje-zalecenia-decyzje.xlsx";
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
		setIncludeRecommendationsInExport(false);
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

			return INSPECTION_EXPORT_COLUMNS.map((column) => column.key).filter(
				(key) => nextSet.has(key),
			);
		});
	};

	const toggleRecommendationExportColumn = (
		columnKey: RecommendationExportColumnKey,
		isSelected: boolean,
	) => {
		setSelectedRecommendationExportColumns((prev) => {
			const nextSet = new Set(prev);
			if (isSelected) {
				nextSet.add(columnKey);
			} else {
				if (prev.length <= 1) {
					return prev;
				}
				nextSet.delete(columnKey);
			}

			return RECOMMENDATION_EXPORT_COLUMNS.map((column) => column.key).filter(
				(key) => nextSet.has(key),
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
			(includeInspectionsInExport &&
				selectedInspectionExportColumns.length === 0) ||
			(includeRecommendationsInExport &&
				selectedRecommendationExportColumns.length === 0) ||
			(includeDecisionsInExport && selectedDecisionExportColumns.length === 0)
		) {
			return;
		}

		const orderedInspectionColumns = INSPECTION_EXPORT_COLUMNS.map(
			(column) => column.key,
		).filter((key) => selectedInspectionExportColumns.includes(key));

		const orderedRecommendationColumns = RECOMMENDATION_EXPORT_COLUMNS.map(
			(column) => column.key,
		).filter((key) => selectedRecommendationExportColumns.includes(key));

		const orderedDecisionColumns = DECISION_EXPORT_COLUMNS.map(
			(column) => column.key,
		).filter((key) => selectedDecisionExportColumns.includes(key));

		setIsExportConfigModalOpen(false);
		void handleExportCurrentView(
			orderedInspectionColumns,
			orderedRecommendationColumns,
			orderedDecisionColumns,
			includeInspectionsInExport,
			includeRecommendationsInExport,
			includeDecisionsInExport,
		);
	};

	const openCreateModal = async () => {
		if (!canManageSanctionRequests) {
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
		if (!canManageSanctionRequests) {
			setError("Konto zewnętrzne ma dostęp tylko do odczytu.");
			return;
		}

		if (!selectedItem || !selectedItem.canEdit) {
			return;
		}

		setEditingItem(selectedItem);
		setForm(requestToForm(selectedItem));
		setFormError(null);
		setShowRequiredFieldErrors(false);
		setVersionConflictUpdatedAt(null);
		setSaveLockConflict(null);
		setIsFormOpen(true);
		await loadInspectionOptions();
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

		const result = await fetchSanctionRequests(operatorLogin, {
			sortBy: "lp",
			sortOrder: "asc",
		});
		if (!result.ok) {
			setFormError(result.error);
			return;
		}

		setItems(result.data.items);
		setTotal(result.data.total);
		const refreshed = result.data.items.find(
			(item) => item.id === editingItem.id,
		);
		if (!refreshed) {
			closeModal();
			return;
		}

		setEditingItem(refreshed);
		setForm(requestToForm(refreshed));
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

		const deletedEntityName =
			selectedItem.nazwaPodmiotuObjetegoInspekcja?.trim() ||
			selectedItem.nazwaPodmiotuObjetegoSankcjaList[0]?.trim() ||
			"";

		setIsDeletingItem(true);
		setError(null);

		const result = await deleteSanctionRequest(operatorLogin, selectedItem.id);
		if (!result.ok) {
			setError(result.error);
			setIsDeletingItem(false);
			return;
		}

		setIsDeleteConfirmModalOpen(false);
		setSelectedId(null);
		await loadItems();
		setDeleteSuccessEntityName(deletedEntityName);
		setIsDeleteSuccessModalOpen(true);
		setIsDeletingItem(false);
	};

	const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
		event.preventDefault();
		if (!canManageSanctionRequests) {
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
		const hasMissingRequiredFields = isRequiredInspectionMissing;

		setShowRequiredFieldErrors(true);

		if (hasMissingRequiredFields) {
			setFormError(null);
			return;
		}

		const payload = formToPayload(form);
		if (!payload) {
			setFormError(
				"Wprowadź poprawne dane: wybierz id inspekcji albo podaj nazwę podmiotu objętego inspekcją.",
			);
			return;
		}

		setShowRequiredFieldErrors(false);

		if (editingItem) {
			const basePayload = formToPayload(requestToForm(editingItem));
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
				? await updateSanctionRequest(operatorLogin, editingItem.id, payload, {
						expectedUpdatedAt: editingItem.zaktualizowanoO,
						lockToken: editRecordLock.lockToken,
					})
				: await createSanctionRequest(operatorLogin, payload);

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
					setVersionConflictUpdatedAt(result.currentUpdatedAt ?? null);
					setFormError(
						"Dane zostały zmienione przez innego użytkownika. Odśwież widok i spróbuj ponownie.",
					);
					return;
				}

				setFormError(result.error);
				return;
			}

			closeModal();
			await loadItems();
			setSelectedId(result.data.id);
			setSuccessEntityName(
				form.nazwaPodmiotuObjetegoInspekcja.trim() ||
					shortenInsuranceEntityName(
						selectedInspectionOption?.nazwaPodmiotu ?? "",
					) ||
					form.nazwaPodmiotuObjetegoSankcjaList[0] ||
					"",
			);
			setSuccessInspectionCode(
				resolveInspectionCode({
					inspectionKod: result.data.inspectionKod,
					kodInspekcji: result.data.kodInspekcji,
					inspectionLp: result.data.inspectionLp,
					inspectionId: result.data.inspectionId,
				}) ||
					selectedInspectionOption?.inspectionCode ||
					"",
			);
			setSuccessMode(wasEditing ? "edit" : "create");
			setIsSuccessModalOpen(true);
		} catch {
			setFormError("Nie udało się zapisać wniosku sankcyjnego.");
		} finally {
			setIsSubmitting(false);
		}
	};

	const isRequiredInspectionMissing =
		showRequiredFieldErrors && !form.isInspectionMissing && !form.inspectionId.trim();

	return (
		<section className="rounded-2xl border border-slate-700/70 bg-[#101f39] p-4 sm:p-5">
			<TablePanelToolbar
				title="Wnioski sankcyjne"
				canClearFilters={canClearFilters}
				canResetColumnWidths={hasCustomColumnWidths}
				isExporting={isExporting}
				hasRowsToExport={
					filteredAndSortedItems.length > 0 &&
					visibleSanctionRequestColumnDefinitions.length > 0
				}
				onOpenViewModal={handleOpenSanctionViewModal}
				onClearFilters={clearFilters}
				onResetColumnWidths={handleResetColumnWidths}
				onExport={handleOpenExportConfigModal}
				actions={
					<>
						{canManageSanctionRequests ? (
							<>
								<button
									type="button"
									onClick={() => void openCreateModal()}
									className="inline-flex h-10 items-center gap-2 rounded-lg border border-[#8ec5a1] bg-[#b9e8c9] px-3.5 font-semibold text-[#1f5130] text-sm transition-colors hover:bg-[#a5debb]"
								>
									<Plus size={15} />
									Dodaj wniosek
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
				<table className="min-w-420 border-collapse text-slate-900 text-sm">
					<TableHeaderWithFilters
						visibleColumns={visibleSanctionRequestColumnDefinitions.map(
							(column) => ({
								...column,
								tooltip: SANCTION_REQUEST_COLUMN_TOOLTIPS[column.key],
							}),
						)}
						sortColumnKey={sortColumnKey}
						sortDirection={sortDirection}
						advancedFilters={advancedFilters}
						columnFilters={columnFilters}
						onSortByColumn={handleSortByColumn}
						onOpenAdvancedFilter={openAdvancedFilterForColumn}
						onFilterChange={handleFilterChange}
						columnWidths={columnWidths}
						onResizeColumn={handleResizeColumn}
						minColumnWidth={SANCTION_REQUESTS_MIN_COLUMN_WIDTH}
						controlsInFilterRow
						wrapHeaderLabels
						truncateWrappedHeaderLabels={false}
					/>
					<tbody>
						{paginatedSanctionRequestItems.map((item) => {
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
									{visibleSanctionRequestColumnDefinitions.map((column) => {
										const value = getCellValue(item, column.key) || "-";
										const inspectionCode =
											column.key === "inspectionId"
												? resolveInspectionCode({
													inspectionKod: item.inspectionKod,
													kodInspekcji: item.kodInspekcji,
													inspectionLp: item.inspectionLp,
													inspectionId: item.inspectionId,
												}).trim()
												: "";
										const hasInspectionLink =
											column.key === "inspectionId" && inspectionCode.length > 0;
										const isListColumn =
											column.key === "nazwaPodmiotuObjetegoSankcjaList" ||
											column.key === "sankcjaList" ||
											column.key === "podstawaPrawnaSankcjiList" ||
											column.key === "naruszeniaSkutkujaceSankcjaList";
										const rawListValues =
											column.key === "nazwaPodmiotuObjetegoSankcjaList"
												? item.nazwaPodmiotuObjetegoSankcjaList
												: column.key === "sankcjaList"
													? item.sankcjaList
													: column.key === "podstawaPrawnaSankcjiList"
														? item.podstawaPrawnaSankcjiList
														: column.key === "naruszeniaSkutkujaceSankcjaList"
															? item.naruszeniaSkutkujaceSankcjaList
															: [];
										const stackedLineValues = isListColumn
											? rawListValues.filter((entry: string) => entry.trim().length > 0)
											: [];

										return (
											<td
												key={column.key}
												className="whitespace-normal break-words px-3 py-2.5 align-top"
											>
												{isListColumn && stackedLineValues.length > 1 ? (
													<div className="space-y-0.5">
														{stackedLineValues.map((entry: string, index: number) => (
															<div key={`${column.key}-${item.id}-${index}`}>{entry}</div>
														))}
													</div>
												) : isListColumn ? (
													stackedLineValues[0] || "-"
												) : hasInspectionLink ? (
													<button
														type="button"
														onClick={(event) => {
															event.stopPropagation();
															openInspectionFromDashboard(inspectionCode);
														}}
														className="cursor-pointer rounded px-1 text-left text-[#1f4f8f] underline decoration-[#9bb8de] underline-offset-2 transition-colors hover:text-[#163a68]"
														title="Przejdź do rejestru Inspekcje i zaznacz ten rekord"
													>
														{value}
													</button>
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
									colSpan={visibleSanctionRequestColumnDefinitions.length}
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
				description="Wnioski sankcyjne eksportują aktualny widok tabeli. Wybierz dane powiązane."
				relationsLabel="Powiąż wybrane wnioski sankcyjne z:"
				relations={[
					{
						id: "inspections",
						label: "Inspekcje",
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
						id: "recommendations",
						label: "Zalecenia",
						enabled: includeRecommendationsInExport,
						selectedCount: selectedRecommendationExportColumns.length,
						onToggle: () => {
							setIncludeRecommendationsInExport((prev) => {
								const next = !prev;
								if (next) {
									setActiveExportColumnsTab("recommendations");
								}
								return next;
							});
						},
					},
					{
						id: "decisions",
						label: "Decyzje zobowiązujące",
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
						id: "recommendations",
						label: "Zalecenia",
						columns: RECOMMENDATION_EXPORT_COLUMNS.map((column) => ({
							key: column.key,
							label: column.label,
						})),
						selectedKeys: selectedRecommendationExportColumns,
						onToggleKey: (key, isSelected) =>
							toggleRecommendationExportColumn(
								key as RecommendationExportColumnKey,
								isSelected,
							),
						onSelectAll: () =>
							setSelectedRecommendationExportColumns(
								RECOMMENDATION_EXPORT_COLUMNS.map((column) => column.key),
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
							toggleDecisionExportColumn(
								key as DecisionExportColumnKey,
								isSelected,
							),
						onSelectAll: () =>
							setSelectedDecisionExportColumns(
								DECISION_EXPORT_COLUMNS.map((column) => column.key),
							),
					},
				]}
				activeTabId={activeExportColumnsTab}
				onActiveTabChange={(tabId) =>
					setActiveExportColumnsTab(
						tabId as "inspections" | "recommendations" | "decisions",
					)
				}
				onClose={() => setIsExportConfigModalOpen(false)}
				onConfirm={handleConfirmExportFromModal}
				isConfirmDisabled={
					isExporting ||
					(includeInspectionsInExport &&
						selectedInspectionExportColumns.length === 0) ||
					(includeRecommendationsInExport &&
						selectedRecommendationExportColumns.length === 0) ||
					(includeDecisionsInExport &&
						selectedDecisionExportColumns.length === 0)
				}
				isExporting={isExporting}
			/>

			<TableAdvancedFilterModal
				isOpen={isAdvancedFilterModalOpen}
				anchor={advancedFilterAnchor}
				columnLabel={
					SANCTION_REQUEST_COLUMNS.find(
						(column) => column.key === advancedFilterColumnKey,
					)?.label ?? "Kolumna"
				}
				searchValue={advancedFilterSearch}
				visibleValues={visibleAdvancedFilterValues}
				selectedValues={selectedAdvancedFilterValues}
				selectedDateRange={selectedAdvancedFilterDateRange}
				onDateRangeChange={setAdvancedFilterDateRange}
				onClose={() => setIsAdvancedFilterModalOpen(false)}
				onSearchChange={setAdvancedFilterSearch}
				onSelectAllVisible={selectAllVisibleAdvancedFilterValues}
				onClearSelectedColumn={clearAdvancedFilterForSelectedColumn}
				onToggleValue={toggleAdvancedFilterValue}
				onClearAllFilters={clearFilters}
			/>

			<TableColumnPickerModal<SanctionRequestColumnKey, never>
				isOpen={isColumnPickerOpen}
				columns={SANCTION_REQUEST_COLUMNS}
				hiddenColumns={draftHiddenColumns}
				visibleColumnsCount={draftVisibleSanctionRequestColumns.length}
				onClose={() => setIsColumnPickerOpen(false)}
				onChangeColumnVisibility={handleDraftColumnVisibilityChange}
				onChangeColumnDisplayMode={(columnKey, value) => {
					if (!isSanctionShortNameColumnKey(columnKey)) {
						return;
					}

					if (value !== "full" && value !== "short") {
						return;
					}

					setDraftSanctionShortNameVariants((prev) => ({
						...prev,
						[columnKey]: value,
					}));
				}}
				columnDisplayModeOptions={columnDisplayModeOptionsByKey}
				columnDisplayModeValues={draftColumnDisplayModeValuesByKey}
				onResetSelection={handleResetSanctionViewSelection}
				onShowAllColumns={handleDraftSelectAllColumns}
				onHideAllColumns={handleDraftDeselectAllColumns}
				onApply={handleApplySanctionViewChanges}
				title="Widok tabeli"
			/>

			<RegistryFormScaffold
				isOpen={isFormOpen}
				title={editingItem ? "Edytuj wniosek sankcyjny" : "Dodaj wniosek sankcyjny"}
				onClose={closeModal}
				onSubmit={(event) => void handleSubmit(event)}
				isContentReadOnly={isReadOnlyDueToLock}
				maxWidthClassName="max-w-5xl"
				closeOnBackdropClick={false}
				headerNotices={
					<>
						{inactivityTimeout.isWarning ? (
							<div className="mt-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-amber-900 text-sm">
								<p className="font-semibold">
									Nie wykryto aktywności. Formularz zostanie zamknięty za{" "}
									<span className="tabular-nums">
										{inactivityTimeout.secondsRemaining}
									</span>{" "}
									s.
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
									Nie możesz teraz edytować tego wpisu, ponieważ jest
									edytowany przez innego użytkownika.
								</p>
								<p className="mt-1">
									Rekord edytuje teraz: {lockOwnerLabel}, od{" "}
									{formatLockStartHourMinute(lockAcquiredAt)}.
								</p>
							</div>
						) : null}

						{isEditMode && editRecordLock.isConnectionLost ? (
							<p className="mt-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 font-medium text-amber-800 text-sm">
								{editRecordLock.error ??
									"Utracono połączenie z serwerem — trwa próba odnowienia blokady..."}
							</p>
						) : null}

						{isEditMode && editRecordLock.isExpired ? (
							<p className="mt-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 font-medium text-rose-800 text-sm">
								{editRecordLock.error ??
									"Czas edycji wygasł — połączenie zostało przerwane zbyt długo. Zamknij formularz i otwórz ponownie."}
							</p>
						) : null}

						{isEditMode && editRecordLock.isAcquireFailed ? (
							<div className="mt-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-rose-800 text-sm">
								<p className="font-medium">
									{editRecordLock.error ??
										"Nie udało się założyć blokady rekordu."}
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
									<div className="text-sm text-slate-700">
										<SingleSelectPortalField
											label="Powiązanie z inspekcją *"
											value={form.inspectionId}
											options={inspectionSelectOptions}
											placeholder={
												isInspectionOptionsLoading
													? "Ładowanie listy inspekcji..."
													: "Wybierz id inspekcji"
											}
											enableSearch
											searchPlaceholder="Wyszukaj id inspekcji..."
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
													nazwaPodmiotuObjetegoInspekcja:
														shortenInsuranceEntityName(
															selectedOption?.nazwaPodmiotu ?? "",
														) ||
														prev.nazwaPodmiotuObjetegoInspekcja,
												}));
											}}
											disabled={
												isReadOnlyDueToLock ||
												form.isInspectionMissing ||
												isInspectionOptionsLoading
											}
										/>
										<label className="mt-2 inline-flex items-center gap-2 font-medium text-slate-700 text-xs">
											<input
												type="checkbox"
												checked={form.isInspectionMissing}
												disabled={isReadOnlyDueToLock}
												onChange={(event) => {
													const checked = event.target.checked;
													setForm((prev) => ({
														...prev,
														isInspectionMissing: checked,
														inspectionId: checked ? "" : prev.inspectionId,
														nazwaPodmiotuObjetegoInspekcja: checked
															? ""
															: prev.nazwaPodmiotuObjetegoInspekcja,
													}));
												}}
											/>
											Brak powiązania z kodem inspekcji
										</label>
										</div>

									{form.isInspectionMissing ? (
											<SingleSelectPortalField
											label="Nazwa podmiotu objętego inspekcją"
											options={nazwaPodmiotuInspekcjaOptions}
											value={form.nazwaPodmiotuObjetegoInspekcja}
											onChange={(next) =>
												setForm((prev) => ({
													...prev,
													nazwaPodmiotuObjetegoInspekcja: next,
												}))
											}
											placeholder="Wybierz podmiot"
											enableSearch
											searchPlaceholder="Wyszukaj podmiot..."
											disabled={isReadOnlyDueToLock}
										/>
									) : (
										<label className="text-sm text-slate-700">
											<span className="mb-1 block">
												Nazwa podmiotu objętego inspekcją
											</span>
											<input
												value={form.nazwaPodmiotuObjetegoInspekcja}
												onChange={(event) =>
													setForm((prev) => ({
														...prev,
														nazwaPodmiotuObjetegoInspekcja: event.target.value,
													}))
												}
												disabled
												className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none transition-colors focus:border-blue-400 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-700"
											/>
										</label>
									)}

									<MultiSelectField
										label="Nazwa podmiotu objętego sankcją"
										options={resolvedNazwaPodmiotuSankcjaSelectOptions}
										values={form.nazwaPodmiotuObjetegoSankcjaList}
										enableSearch
										searchPlaceholder="Wyszukaj podmiot objęty sankcją..."
										disabled={isReadOnlyDueToLock}
										onChange={(next) =>
											setForm((prev) => ({
												...prev,
												nazwaPodmiotuObjetegoSankcjaList: next,
											}))
										}
									/>

									<DateInputWithCalendar
										label="Data wniosku"
										value={form.dataWniosku}
										onChange={(next) =>
											setForm((prev) => ({
												...prev,
												dataWniosku: next,
											}))
										}
										disabled={isReadOnlyDueToLock}
									/>

									<SingleSelectPortalField
										label="Wniosek do"
										options={wniosekDoOptions}
										value={form.wniosekDo}
										placeholder="Wybierz"
										onChange={(next) =>
											setForm((prev) => ({ ...prev, wniosekDo: next }))
										}
										disabled={isReadOnlyDueToLock}
									/>

									<MultiSelectField
										label="Sankcja"
										options={sankcjaOptions}
										values={form.sankcjaList}
										disabled={isReadOnlyDueToLock}
										onChange={(next) =>
											setForm((prev) => ({ ...prev, sankcjaList: next }))
										}
									/>

									<MultiSelectField
										label="Podstawa prawna sankcji"
										options={podstawaPrawnaOptions}
										values={form.podstawaPrawnaSankcjiList}
										disabled={isReadOnlyDueToLock}
										onChange={(next) =>
											setForm((prev) => ({
												...prev,
												podstawaPrawnaSankcjiList: next,
											}))
										}
									/>

									<MultiSelectField
										label="Naruszenia skutkujące sankcją"
										options={naruszeniaOptions}
										values={form.naruszeniaSkutkujaceSankcjaList}
										disabled={isReadOnlyDueToLock}
										onChange={(next) =>
											setForm((prev) => ({
												...prev,
												naruszeniaSkutkujaceSankcjaList: next,
											}))
										}
									/>

									<SingleSelectPortalField
										label="Informacja o wszczęciu postępowania"
										options={informacjaOptions}
										value={form.czyMamyInformacjeOWszczeciuPostepowania}
										placeholder="Wybierz"
										onChange={(next) =>
											setForm((prev) => ({
												...prev,
												czyMamyInformacjeOWszczeciuPostepowania: next,
											}))
										}
										disabled={isReadOnlyDueToLock}
									/>

									<SingleSelectPortalField
										label="Rozstrzygnięcie"
										options={rozstrzygniecieOptions}
										value={form.rozstrzygniecie}
										placeholder="Wybierz"
										onChange={(next) =>
											setForm((prev) => ({ ...prev, rozstrzygniecie: next }))
										}
										disabled={isReadOnlyDueToLock}
									/>

									<label className="text-sm text-slate-700 sm:col-span-2">
										<span className="mb-1 block">Komentarz</span>
										<textarea
											rows={2}
											value={form.komentarz}
											disabled={isReadOnlyDueToLock}
											onChange={(event) =>
												setForm((prev) => ({
													...prev,
													komentarz: event.target.value,
												}))
											}
											className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none transition-colors focus:border-blue-400 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-700"
										/>
									</label>
				</div>
			</RegistryFormScaffold>

			<SanctionRequestsSuccessModal
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
				heading="Wniosek sankcyjny został usunięty"
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
						aria-label="Zamknij potwierdzenie usunięcia wniosku sankcyjnego"
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
						aria-label="Potwierdzenie usunięcia wniosku sankcyjnego"
						className="relative z-10 w-full max-w-lg rounded-2xl border border-slate-300 bg-white p-5 text-slate-900 shadow-[0_24px_56px_rgba(2,8,23,0.35)]"
					>
						<h3 className="font-semibold text-base text-slate-900">
							Usuń wniosek sankcyjny
						</h3>
						<p className="mt-2 text-slate-700 text-sm">
							Czy usunąć wniosek sankcyjny?
						</p>

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
