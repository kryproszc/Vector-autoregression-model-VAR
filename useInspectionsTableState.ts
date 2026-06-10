"use client";

import { Pencil, Plus, Trash2 } from "lucide-react";
import {
	type SetStateAction,
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
} from "react";

import type { AuthRole } from "@/app/_components/home-tabs/types";
import { fetchDictionaryEntries } from "@/features/dictionaries/api";
import { INSPECTION_VIEW_OPTIONS } from "@/features/inspections/data";
import {
	type AddInspectionForm,
	type DictionarySelectOption,
	type InspectionListResponse,
	type InspectionPeopleOption,
	type RawInspectionRow,
	getBaseInspectionColumnKeys,
	getInspectionApiErrorMessage,
	getUserDisplayName,
	joinMultiValueField,
	mapDictionaryEntriesToOptions,
	mapDictionaryEntriesToSelectOptions,
	mapRowToAddForm,
	normalizeInspectionRow,
	parseMultiValueField,
} from "@/features/inspections/components/inspections-panel.utils";
import { InspectionsDataTable } from "@/features/inspections/components/inspections-panel/InspectionsDataTable";
import { InspectionsFormModal } from "@/features/inspections/components/inspections-panel/InspectionsFormModal";
import { useInspectionsTableState } from "@/features/inspections/hooks/useInspectionsTableState";
import { fetchObligatingDecisions } from "@/features/obligating-decisions/api";
import { fetchRecommendations } from "@/features/recommendations/api";
import { fetchSanctionRequests } from "@/features/sanction-requests/api";
import { EntitySuccessModal } from "@/shared/components/EntitySuccessModal";
import { ExportConfigModal } from "@/shared/components/export/ExportConfigModal";
import { TableAdvancedFilterModal } from "@/shared/components/table/TableAdvancedFilterModal";
import { TableColumnPickerModal } from "@/shared/components/table/TableColumnPickerModal";
import { TablePagination } from "../../../shared/components/table/TablePagination";
import { TablePanelToolbar } from "@/shared/components/table/TablePanelToolbar";
import { useInactivityTimeout } from "@/shared/hooks/useInactivityTimeout";
import { useRecordLock } from "@/shared/hooks/useRecordLock";

const INACTIVITY_TIMEOUT_MS = 60_000; // 1 minuta (do testów)
const INACTIVITY_WARNING_MS = 30_000; // 30 sekund ostrzeżenia
const TABLE_PAGE_SIZE_OPTIONS = [20, 30, 50, 70, 100];
const INSPECTIONS_COLUMN_WIDTHS_STORAGE_PREFIX =
	"triangle.ui.inspections.column-widths";
const INSPECTIONS_NAME_VARIANTS_STORAGE_PREFIX =
	"triangle.ui.inspections.name-variants";
const INSPECTIONS_TABLE_VIEW_STORAGE_PREFIX =
	"triangle.ui.inspections.table-view";
const INSPECTIONS_DATE_COLUMN_WIDTH = 190;
const INSPECTIONS_EXTENDED_DATE_COLUMN_WIDTH = 202;
const INSPECTIONS_SCOPE_DETAILS_COLUMN_WIDTH = 320;
const DEFAULT_INSPECTIONS_COLUMN_WIDTHS: Partial<
	Record<InspectionColumnKey, number>
> = {
	lp: 90,
	kodInspekcji: 170,
	nazwaPodmiotu: 200,
	typInspekcji: 170,
	zakresInspekcji: 280,
	szczegolyDotyczaceZakresu: INSPECTIONS_SCOPE_DETAILS_COLUMN_WIDTH,
	aspektKonsumencki: 190,
	poczatekInspekcji: INSPECTIONS_DATE_COLUMN_WIDTH,
	koniecInspekcji: INSPECTIONS_DATE_COLUMN_WIDTH,
	osobaKierujaca: 260,
	skladZespolu: 260,
	rynek: 190,
	rodzajPodmiotu: 210,
	dataProtokolu: INSPECTIONS_DATE_COLUMN_WIDTH,
	dataDoreczeniaProtokolu: INSPECTIONS_DATE_COLUMN_WIDTH,
	dataAkceptacjiSprawozdania: INSPECTIONS_EXTENDED_DATE_COLUMN_WIDTH,
	dataDoreczeniaPisma: INSPECTIONS_DATE_COLUMN_WIDTH,
	dataPismaZastrzezenia: INSPECTIONS_EXTENDED_DATE_COLUMN_WIDTH,
	dataWyslaniaPismaZZastrzezeniami: INSPECTIONS_EXTENDED_DATE_COLUMN_WIDTH,
	dataWplywuPisma: INSPECTIONS_EXTENDED_DATE_COLUMN_WIDTH,
	dataPismaZOdpowiedzia: INSPECTIONS_EXTENDED_DATE_COLUMN_WIDTH,
	dataWyslaniaPismaZOdpowiedzia: INSPECTIONS_EXTENDED_DATE_COLUMN_WIDTH,
	dataAkceptacjiNoty: INSPECTIONS_DATE_COLUMN_WIDTH,
	dataZalecen: INSPECTIONS_DATE_COLUMN_WIDTH,
	status: 190,
	komentarz: 240,
};
const INSPECTIONS_MIN_COLUMN_WIDTH = 90;
import type {
	InspectionColumnKey,
	InspectionRow,
	InspectionViewId,
} from "@/features/inspections/types";
import {
	addWorksheetWithStyles,
	createStyledExportWorkbook,
	saveWorkbookAsXlsx,
} from "@/shared/utils/excel-export";
import { toDateInputValue, toDateList } from "@/shared/utils/date";


const INSPECTIONS_API_URL = "/api/structure/inspections";
const RECOMMENDATIONS_AVAILABLE_INSPECTIONS_API_URL =
	"/api/recommendations/available-inspections";
const SANCTIONS_AVAILABLE_INSPECTIONS_API_URL =
	"/api/sanction-requests/available-inspections";

const DEFAULT_ADD_INSPECTION_FORM: AddInspectionForm = {
	nazwaPodmiotu: "",
	typInspekcji: "",
	zakresInspekcji: "",
	szczegolyDotyczaceZakresu: "",
	aspektKonsumencki: "NIE",
	poczatekInspekcji: "",
	koniecInspekcji: "",
	osobaKierujaca: "",
	skladZespolu: "",
	rynek: "",
	rodzajPodmiotu: "",
	dataProtokolu: "",
	dataDoreczeniaProtokolu: "",
	dataAkceptacjiSprawozdania: "",
	dataDoreczeniaPisma: "",
	dataPismaZastrzezenia: "",
	dataWyslaniaPismaZZastrzezeniami: "",
	dataWplywuPisma: "",
	dataPismaZOdpowiedzia: "",
	dataWyslaniaPismaZOdpowiedzia: "",
	dataAkceptacjiNoty: "",
	dataZalecen: "",
	status: "",
	komentarz: "",
	brakDataDoreczeniaPisma: false,
	brakDataPismaZastrzezenia: false,
	brakDataWyslaniaPismaZZastrzezeniami: false,
	brakDataWplywuPisma: false,
	brakDataPismaZOdpowiedzia: false,
	brakDataWyslaniaPismaZOdpowiedzia: false,
};

function normalizeInspectionScopeValues(values: string[]) {
	const normalized = values
		.flatMap((value) => value.split(";"))
		.map((value) => value.trim())
		.filter(Boolean)
		.filter((value) => {
			const lowered = value.toLowerCase();
			return lowered !== "brak" && lowered !== "-";
		});

	return Array.from(new Set(normalized)).sort((left, right) =>
		left.localeCompare(right, "pl", { sensitivity: "base" }),
	);
}

type InspectionsPanelProps = {
	operatorLogin: string;
	authRole: AuthRole;
	isObserver?: boolean;
};

type InspectionLockConflict = {
	ownerLogin: string;
	ownerDisplayName: string;
	acquiredAt: string;
};

type InspectionDomainError = {
	code: string;
	detail: string;
	memberUserId: number | null;
};

type InspectionStatusValidationViolation = {
	violationCodeId: number | null;
	message: string;
};

type InspectionNoLetterFlags = {
	brakDataDoreczeniaPisma: boolean;
	brakDataPismaZastrzezenia: boolean;
	brakDataWyslaniaPismaZZastrzezeniami: boolean;
	brakDataWplywuPisma: boolean;
	brakDataPismaZOdpowiedzia: boolean;
	brakDataWyslaniaPismaZOdpowiedzia: boolean;
};

type InspectionNoAcceptanceDatesFlags = {
	brakDatAkceptacjiNoty: boolean;
};

type InspectionNameVariant = "full" | "short" | "user";

type InspectionNameVariantColumnKey =
	| "nazwaPodmiotu"
	| "typInspekcji"
	| "zakresInspekcji"
	| "rodzajPodmiotu"
	| "status";

type InspectionNameVariantByColumn = Record<
	InspectionNameVariantColumnKey,
	InspectionNameVariant
>;

type InspectionShortValuesByColumn = Partial<
	Record<InspectionNameVariantColumnKey, string>
>;

const INSPECTION_NAME_VARIANT_COLUMN_KEYS: InspectionNameVariantColumnKey[] = [
	"nazwaPodmiotu",
	"typInspekcji",
	"zakresInspekcji",
	"rodzajPodmiotu",
	"status",
];

const INSPECTION_NAME_VARIANT_OPTIONS = [
	{ value: "full", label: "Nazwa pełna" },
	{ value: "short", label: "Nazwa skrócona" },
] as const;

const DEFAULT_INSPECTION_NAME_VARIANTS: InspectionNameVariantByColumn = {
	nazwaPodmiotu: "short",
	typInspekcji: "full",
	zakresInspekcji: "full",
	rodzajPodmiotu: "full",
	status: "full",
};

function isInspectionNameVariantColumnKey(
	columnKey: InspectionColumnKey,
): columnKey is InspectionNameVariantColumnKey {
	return INSPECTION_NAME_VARIANT_COLUMN_KEYS.includes(
		columnKey as InspectionNameVariantColumnKey,
	);
}

function isInspectionNameVariant(value: unknown): value is InspectionNameVariant {
	return value === "full" || value === "short" || value === "user";
}

function isInspectionNameVariantAllowedForColumn(
	columnKey: InspectionNameVariantColumnKey,
	value: InspectionNameVariant,
) {
	void columnKey;
	return value === "full" || value === "short";
}

const RECOMMENDATIONS_CHANGED_EVENT = "recommendations:changed";
const INSPECTIONS_CHANGED_EVENT = "inspections:changed";
const DASHBOARD_OPEN_INSPECTION_EVENT = "dashboard:open-inspection";
const DASHBOARD_OPEN_INSPECTION_CODE_KEY = "triangle.dashboard.openInspectionCode";

function resolveInspectionLockRecordIds(
	rawRow: RawInspectionRow,
	fallbackId: string,
) {
	const rawId = String(rawRow.id ?? "").trim();
	const rawLp = String(rawRow.lp ?? "").trim();
	const rawInspectionKod = String(
		(rawRow as { inspectionKod?: unknown }).inspectionKod ??
			(rawRow as { kodInspekcji?: unknown }).kodInspekcji ??
			"",
	).trim();

	const candidates = [rawId, rawLp, rawInspectionKod, fallbackId]
		.map((value) => value.trim())
		.filter(Boolean);

	return Array.from(new Set(candidates));
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

async function readInspectionDomainError(
	response: Response,
): Promise<InspectionDomainError | null> {
	const contentType = response.headers.get("content-type") ?? "";
	if (!contentType.includes("application/json")) {
		return null;
	}

	try {
		const payload = (await response.clone().json()) as Record<string, unknown>;
		const detailSource = payload.detail;
		const detailObject =
			detailSource && typeof detailSource === "object" && !Array.isArray(detailSource)
				? (detailSource as Record<string, unknown>)
				: payload;
		const code =
			typeof detailObject.code === "string"
				? detailObject.code.trim().toUpperCase()
				: typeof payload.code === "string"
					? payload.code.trim().toUpperCase()
					: "";
		const detail =
			typeof detailObject.detail === "string"
				? detailObject.detail.trim()
				: typeof detailObject.message === "string"
					? detailObject.message.trim()
					: typeof payload.detail === "string"
						? payload.detail.trim()
						: "";
		const memberCandidate =
			detailObject.memberUserId ?? detailObject.member_user_id ?? detailObject.userId;
		const memberNumeric =
			typeof memberCandidate === "number"
				? memberCandidate
				: typeof memberCandidate === "string"
					? Number(memberCandidate.trim())
					: NaN;

		if (!code) {
			return null;
		}

		return {
			code,
			detail,
			memberUserId:
				Number.isFinite(memberNumeric) && memberNumeric > 0 ? memberNumeric : null,
		};
	} catch {
		return null;
	}
}

function mapInspectionStatusViolationMessage(violationCodeId: number | null) {
	switch (violationCodeId) {
		case 1001:
			return "Do tej kontroli nie dodano jeszcze zalecenia. Najpierw dodaj zalecenie, a potem ustaw ten status.";
		case 1002:
			return "Do tej kontroli zostało już dodane zalecenie. Ten status można ustawić tylko wtedy, gdy kontrola nie ma żadnych zaleceń.";
		case 1003:
			return "Do tej kontroli nie dodano jeszcze wniosku sankcyjnego. Najpierw dodaj wniosek sankcyjny, a potem ustaw ten status.";
		case 1004:
			return "Do tej kontroli został już dodany wniosek sankcyjny. Ten status można ustawić tylko wtedy, gdy kontrola nie ma żadnych wniosków sankcyjnych.";
		default:
			return "Nie można zapisać rekordu z powodu niespełnionych relacji dla wybranego statusu.";
	}
}

async function readInspectionStatusValidationViolations(
	response: Response,
): Promise<InspectionStatusValidationViolation[] | null> {
	const contentType = response.headers.get("content-type") ?? "";
	if (!contentType.includes("application/json")) {
		return null;
	}

	try {
		const payload = (await response.clone().json()) as Record<string, unknown>;
		const detailSource = payload.detail;
		const detailObject =
			detailSource &&
			typeof detailSource === "object" &&
			!Array.isArray(detailSource)
				? (detailSource as Record<string, unknown>)
				: payload;

		const code =
			typeof detailObject.code === "string"
				? detailObject.code.trim().toUpperCase()
				: typeof payload.code === "string"
					? payload.code.trim().toUpperCase()
					: "";
		const codeIdCandidate = detailObject.codeId ?? payload.codeId;
		const codeId =
			typeof codeIdCandidate === "number"
				? codeIdCandidate
				: typeof codeIdCandidate === "string"
					? Number(codeIdCandidate.trim())
					: NaN;

		const isStatusValidationError =
			code === "INSPECTION_STATUS_RELATIONS_VALIDATION_FAILED" ||
			(Number.isFinite(codeId) && codeId === 1100);
		if (!isStatusValidationError) {
			return null;
		}

		const violationsSource =
			detailObject.violations ?? payload.violations ?? detailObject.items ?? payload.items;
		const violationItems = Array.isArray(violationsSource) ? violationsSource : [];

		const parsedViolations = violationItems
			.map((item) => {
				if (!item || typeof item !== "object") {
					return null;
				}

				const source = item as Record<string, unknown>;
				const violationCodeCandidate =
					source.violationCodeId ?? source.codeId ?? source.code;
				const numericViolationCode =
					typeof violationCodeCandidate === "number"
						? violationCodeCandidate
						: typeof violationCodeCandidate === "string"
							? Number(violationCodeCandidate.trim())
							: NaN;
				const normalizedViolationCode = Number.isFinite(numericViolationCode)
					? numericViolationCode
					: null;

				const detailMessage =
					typeof source.detail === "string"
						? source.detail.trim()
						: typeof source.message === "string"
							? source.message.trim()
							: "";
				const mappedMessage =
					normalizedViolationCode === 1001 ||
					normalizedViolationCode === 1002 ||
					normalizedViolationCode === 1003 ||
					normalizedViolationCode === 1004
						? mapInspectionStatusViolationMessage(normalizedViolationCode)
						: "";

				return {
					violationCodeId: normalizedViolationCode,
					message:
						mappedMessage ||
						detailMessage ||
						mapInspectionStatusViolationMessage(normalizedViolationCode),
				};
			})
			.filter(
				(
					violation,
				): violation is InspectionStatusValidationViolation => violation !== null,
			);

		if (parsedViolations.length > 0) {
			return parsedViolations;
		}

		return [
			{
				violationCodeId: null,
				message: mapInspectionStatusViolationMessage(null),
			},
		];
	} catch {
		return null;
	}
}

type RecommendationExportColumnKey =
	| "lp"
	| "kodZalecenia"
	| "inspectionLp"
	| "nazwaPodmiotu"
	| "pozycja"
	| "dataZalecen"
	| "terminyWykonaniaZalecenList"
	| "dataAkceptacjiNotyWeryfikacjiList"
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

const RECOMMENDATION_EXPORT_COLUMNS: ExportColumnDefinition<RecommendationExportColumnKey>[] =
	[
		{ key: "lp", label: "Lp. zalecenia" },
		{ key: "kodZalecenia", label: "Id zalecenia" },
		{ key: "inspectionLp", label: "Id inspekcji" },
		{ key: "nazwaPodmiotu", label: "Nazwa podmiotu" },
		{ key: "pozycja", label: "Liczba zaleceń" },
		{ key: "dataZalecen", label: "Data zaleceń" },
		{ key: "terminyWykonaniaZalecenList", label: "Termin wykonania zaleceń" },
		{
			key: "dataAkceptacjiNotyWeryfikacjiList",
			label: "Data akceptacji noty z weryfikacji",
		},
		{ key: "status", label: "Status" },
		{ key: "komentarz", label: "Komentarz" },
	];

const SANCTION_EXPORT_COLUMNS: ExportColumnDefinition<SanctionExportColumnKey>[] =
	[
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

const DECISION_EXPORT_COLUMNS: ExportColumnDefinition<DecisionExportColumnKey>[] =
	[
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
		{
			key: "osobyProwadzaceIInstancjeList",
			label: "Osoby prowadzące I instancję",
		},
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
		{
			key: "osobyProwadzaceIIInstancjeList",
			label: "Osoby prowadzące II instancję",
		},
		{ key: "dataDecyzjiIIInstancji", label: "Data decyzji II instancji" },
		{
			key: "dataDoreczeniaDecyzjiIIInstancji",
			label: "Data doręczenia decyzji II instancji",
		},
		{ key: "rozstrzygniecieII", label: "Rozstrzygnięcie decyzji II instancji" },
		{ key: "komentarz", label: "Komentarz" },
	];

export function InspectionsPanel({
	operatorLogin,
	authRole,
	isObserver,
}: InspectionsPanelProps) {
	const [inspectionRows, setInspectionRows] = useState<InspectionRow[]>([]);
	const [inspectionShortValuesByRowId, setInspectionShortValuesByRowId] =
		useState<Record<string, InspectionShortValuesByColumn>>({});
	const [inspectionNameVariants, setInspectionNameVariants] =
		useState<InspectionNameVariantByColumn>(DEFAULT_INSPECTION_NAME_VARIANTS);
	const [draftInspectionNameVariants, setDraftInspectionNameVariants] =
		useState<InspectionNameVariantByColumn>(DEFAULT_INSPECTION_NAME_VARIANTS);
	const [areNameVariantsHydrated, setAreNameVariantsHydrated] =
		useState(false);
	const [selectedInspectionId, setSelectedInspectionId] = useState<
		string | null
	>(null);
	const [flashInspectionId, setFlashInspectionId] = useState<string | null>(null);
	const [pendingDashboardInspectionCode, setPendingDashboardInspectionCode] =
		useState<string | null>(null);
	const [isAddModalOpen, setIsAddModalOpen] = useState(false);
	const [isPreviewMode, setIsPreviewMode] = useState(false);
	const [isTeamPickerOpen, setIsTeamPickerOpen] = useState(false);
	const [editingInspectionId, setEditingInspectionId] = useState<string | null>(
		null,
	);
	const [addInspectionForm, setAddInspectionForm] = useState<AddInspectionForm>(
		DEFAULT_ADD_INSPECTION_FORM,
	);
	const [addInspectionError, setAddInspectionError] = useState<string | null>(
		null,
	);
	const [showRequiredInspectionFieldErrors, setShowRequiredInspectionFieldErrors] =
		useState(false);
	const [rowsError, setRowsError] = useState<string | null>(null);
	const [isRowsLoading, setIsRowsLoading] = useState(true);
	const [isSubmittingInspection, setIsSubmittingInspection] = useState(false);
	const [isCreateSuccessModalOpen, setIsCreateSuccessModalOpen] =
		useState(false);
	const [createSuccessEntityName, setCreateSuccessEntityName] = useState("");
	const [createSuccessMode, setCreateSuccessMode] = useState<"create" | "edit">(
		"create",
	);
	const [isExporting, setIsExporting] = useState(false);
	const [isExportConfigModalOpen, setIsExportConfigModalOpen] = useState(false);
	const [includeRecommendationsInExport, setIncludeRecommendationsInExport] =
		useState(false);
	const [includeSanctionsInExport, setIncludeSanctionsInExport] =
		useState(false);
	const [includeDecisionsInExport, setIncludeDecisionsInExport] =
		useState(false);
	const [activeExportColumnsTab, setActiveExportColumnsTab] = useState<
		"recommendations" | "sanctions" | "decisions"
	>("recommendations");
	const [
		selectedRecommendationExportColumns,
		setSelectedRecommendationExportColumns,
	] = useState<RecommendationExportColumnKey[]>(
		RECOMMENDATION_EXPORT_COLUMNS.map((column) => column.key),
	);
	const [selectedSanctionExportColumns, setSelectedSanctionExportColumns] =
		useState<SanctionExportColumnKey[]>(
			SANCTION_EXPORT_COLUMNS.map((column) => column.key),
		);
	const [selectedDecisionExportColumns, setSelectedDecisionExportColumns] =
		useState<DecisionExportColumnKey[]>(
			DECISION_EXPORT_COLUMNS.map((column) => column.key),
		);
	const [entityNameOptions, setEntityNameOptions] = useState<
		DictionarySelectOption[]
	>([]);
	const [inspectionTypeOptions, setInspectionTypeOptions] = useState<string[]>(
		[],
	);
	const [inspectionScopeOptions, setInspectionScopeOptions] = useState<
		DictionarySelectOption[]
	>([]);
	const [inspectionScopeMapByValue, setInspectionScopeMapByValue] =
		useState<Record<string, string>>({});
	const [marketOptions, setMarketOptions] = useState<string[]>([]);
	const [entityTypeOptions, setEntityTypeOptions] = useState<string[]>([]);
	const [inspectionStatusOptions, setInspectionStatusOptions] = useState<
		DictionarySelectOption[]
	>([]);
	const [allUsers, setAllUsers] = useState<InspectionPeopleOption[]>([]);
	const [activeUsers, setActiveUsers] = useState<InspectionPeopleOption[]>([]);
	const [selectedInspectionScopes, setSelectedInspectionScopes] = useState<
		string[]
	>([]);
	const [selectedTeamMemberIds, setSelectedTeamMemberIds] = useState<number[]>(
		[],
	);
	const [teamMemberScopeError, setTeamMemberScopeError] = useState<string | null>(
		null,
	);
	const [outOfScopeTeamMemberUserId, setOutOfScopeTeamMemberUserId] = useState<
		number | null
	>(null);
	const [operatorUserId, setOperatorUserId] = useState<number | null>(null);
	const [operatorTeamId, setOperatorTeamId] = useState<number | null>(null);
	const [selectedLeaderUserId, setSelectedLeaderUserId] = useState<
		number | null
	>(null);
	const [inspectionLeaderUserIdByRowId, setInspectionLeaderUserIdByRowId] =
		useState<Record<string, number | null>>({});
	const [inspectionTeamMemberIdsByRowId, setInspectionTeamMemberIdsByRowId] =
		useState<Record<string, number[]>>({});
	const [
		inspectionAcceptanceDatesByRowId,
		setInspectionAcceptanceDatesByRowId,
	] = useState<Record<string, string[]>>({});
	const [inspectionNoAcceptanceDatesByRowId, setInspectionNoAcceptanceDatesByRowId] =
		useState<Record<string, InspectionNoAcceptanceDatesFlags>>({});
	const [inspectionNoLetterFlagsByRowId, setInspectionNoLetterFlagsByRowId] =
		useState<Record<string, InspectionNoLetterFlags>>({});
	const [inspectionCanEditByRowId, setInspectionCanEditByRowId] = useState<
		Record<string, boolean>
	>({});
	const [inspectionLockRecordIdsByRowId, setInspectionLockRecordIdsByRowId] =
		useState<Record<string, string[]>>({});
	const [inspectionUpdatedAtByRowId, setInspectionUpdatedAtByRowId] = useState<
		Record<string, string | null>
	>({});
	const [versionConflictUpdatedAt, setVersionConflictUpdatedAt] = useState<
		string | null
	>(null);
	const [statusValidationViolations, setStatusValidationViolations] = useState<
		InspectionStatusValidationViolation[]
	>([]);
	const [isStatusValidationModalOpen, setIsStatusValidationModalOpen] =
		useState(false);
	const [saveLockConflict, setSaveLockConflict] =
		useState<InspectionLockConflict | null>(null);
	const [operatorDisplayName, setOperatorDisplayName] = useState(
		operatorLogin.trim(),
	);
	const [dataAkceptacjiNotyList, setDataAkceptacjiNotyList] = useState<
		string[]
	>([]);
	const [isDataAkceptacjiNotyBrak, setIsDataAkceptacjiNotyBrak] =
		useState(false);
	const [didToggleDataAkceptacjiNotyBrak, setDidToggleDataAkceptacjiNotyBrak] =
		useState(false);
	const [isDeleteConfirmModalOpen, setIsDeleteConfirmModalOpen] =
		useState(false);
	const [isDeletingInspection, setIsDeletingInspection] = useState(false);
	const [isDeleteSuccessModalOpen, setIsDeleteSuccessModalOpen] =
		useState(false);
	const [deleteSuccessModalMessage, setDeleteSuccessModalMessage] = useState<
		string | null
	>(null);
	const [columnWidths, setColumnWidths] = useState<
		Partial<Record<InspectionColumnKey, number>>
	>(DEFAULT_INSPECTIONS_COLUMN_WIDTHS);
	const [areColumnWidthsHydrated, setAreColumnWidthsHydrated] = useState(false);
	const canManageInspections = authRole !== "external_user" && !isObserver;
	const isDirector = authRole === "director";
	const normalizedOperatorLogin = operatorLogin.trim().toLowerCase();
	const columnWidthsStorageKey = `${INSPECTIONS_COLUMN_WIDTHS_STORAGE_PREFIX}.${normalizedOperatorLogin}`;
	const nameVariantsStorageKey = `${INSPECTIONS_NAME_VARIANTS_STORAGE_PREFIX}.${normalizedOperatorLogin}`;
	const tableViewStorageKey = `${INSPECTIONS_TABLE_VIEW_STORAGE_PREFIX}.${normalizedOperatorLogin}`;

	const selectedInspectionRow = useMemo(
		() =>
			selectedInspectionId
				? (inspectionRows.find((row) => row.id === selectedInspectionId) ??
					null)
				: null,
		[inspectionRows, selectedInspectionId],
	);

	const selectedInspectionCanEdit = useMemo(
		() =>
			selectedInspectionId
				? (inspectionCanEditByRowId[selectedInspectionId] ?? false)
				: false,
		[inspectionCanEditByRowId, selectedInspectionId],
	);

	const previewInspectionCanEdit = useMemo(
		() =>
			editingInspectionId
				? (inspectionCanEditByRowId[editingInspectionId] ?? false)
				: false,
		[editingInspectionId, inspectionCanEditByRowId],
	);

	const currentEditingInspectionLeaderUserId = useMemo(
		() =>
			editingInspectionId
				? (inspectionLeaderUserIdByRowId[editingInspectionId] ?? null)
				: null,
		[editingInspectionId, inspectionLeaderUserIdByRowId],
	);

	const isEditMode = Boolean(editingInspectionId);
	const canChangeLeaderInEdit =
		authRole === "director" || authRole === "team_lead";
	const canChangeLeaderSelection = !isEditMode || canChangeLeaderInEdit;
	const editingInspectionLockRecordIds = editingInspectionId
		? (inspectionLockRecordIdsByRowId[editingInspectionId] ?? [
				editingInspectionId,
			])
		: [];
	const primaryEditingInspectionLockRecordId =
		editingInspectionLockRecordIds[0] ?? null;
	const alternateEditingInspectionLockRecordIds = useMemo(
		() => editingInspectionLockRecordIds.slice(1),
		[editingInspectionLockRecordIds.join("|")],
	);
	const editInspectionLock = useRecordLock({
		enabled: isAddModalOpen && isEditMode && !isPreviewMode,
		module: "inspections",
		recordId: primaryEditingInspectionLockRecordId,
		alternateRecordIds: alternateEditingInspectionLockRecordIds,
		operatorLogin,
		heartbeatIntervalMs: 20_000,
	});
	const shouldShowLockedByOtherUser =
		Boolean(saveLockConflict) || editInspectionLock.isBlocked;
	const isReadOnlyDueToLock = isEditMode && shouldShowLockedByOtherUser;
	const lockOwnerDisplayName =
		saveLockConflict?.ownerDisplayName ||
		editInspectionLock.owner?.displayName ||
		"";
	const lockOwnerLogin =
		saveLockConflict?.ownerLogin || editInspectionLock.owner?.login || "";
	const lockOwnerLabel =
		lockOwnerDisplayName || lockOwnerLogin
			? `${lockOwnerDisplayName || "Nieznany użytkownik"}${
					lockOwnerLogin ? ` (${lockOwnerLogin})` : ""
				}`
			: "inny użytkownik";
	const lockAcquiredAt =
		saveLockConflict?.acquiredAt ||
		editInspectionLock.lockDetails?.acquiredAt ||
		null;
	const selectedStatusForValidation = addInspectionForm.status.trim();

	const closeInspectionFormModalRef = useRef<() => void>(() => {});
	const inactivityTimeout = useInactivityTimeout({
		enabled: isAddModalOpen,
		inactivityMs: INACTIVITY_TIMEOUT_MS,
		warningMs: INACTIVITY_WARNING_MS,
		onTimeout: () => closeInspectionFormModalRef.current(),
	});

	const inspectionLockNotice = shouldShowLockedByOtherUser
		? `Nie możesz teraz edytować tego wpisu, ponieważ jest edytowany przez innego użytkownika. Rekord edytuje teraz: ${lockOwnerLabel}, od ${formatLockStartHourMinute(lockAcquiredAt)}.`
		: isEditMode && editInspectionLock.isConnectionLost
				? (editInspectionLock.error ??
					"Utracono połączenie z serwerem — trwa próba odnowienia blokady...")
				: isEditMode && editInspectionLock.isExpired
					? (editInspectionLock.error ??
						"Czas edycji wygasł — połączenie zostało przerwane zbyt długo. Zamknij formularz i otwórz ponownie.")
					: isEditMode && editInspectionLock.isAcquireFailed
						? (editInspectionLock.error ??
							"Nie udało się założyć blokady rekordu.")
						: null;

	const loadInspections = useCallback(async () => {
		setRowsError(null);
		setIsRowsLoading(true);

		try {
			const [response, recommendationsResult] = await Promise.all([
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
			]);

			const recommendationDatesByInspectionId = new Map<number, Set<string>>();
			if (recommendationsResult.ok) {
				for (const recommendation of recommendationsResult.data.items) {
					const inspectionId = recommendation.inspectionId;
					const recommendationDates = [
						...toDateList(recommendation.dataZalecen),
					];

					if (
						typeof inspectionId !== "number" ||
						!Number.isFinite(inspectionId) ||
						inspectionId <= 0 ||
						recommendationDates.length === 0
					) {
						continue;
					}

					const existing =
						recommendationDatesByInspectionId.get(inspectionId) ?? new Set<string>();
					for (const dateValue of recommendationDates) {
						existing.add(dateValue);
					}
					recommendationDatesByInspectionId.set(inspectionId, existing);
				}
			}

			if (!response.ok) {
				const apiMessage = await getInspectionApiErrorMessage(
					response,
					"Nie udało się pobrać danych",
				);
				throw new Error(apiMessage);
			}

			const payload = (await response.json()) as unknown;
			const rawItems: unknown[] = Array.isArray(payload)
				? payload
				: Array.isArray((payload as Partial<InspectionListResponse>).items)
					? ((payload as Partial<InspectionListResponse>).items ?? [])
					: [];

			const items = rawItems.map((row, index) => {
				const normalized = normalizeInspectionRow(
					(row ?? {}) as RawInspectionRow,
					index,
				);
				const rowInspectionId = Number((row as { id?: unknown }).id);
				const recommendationDates = Number.isFinite(rowInspectionId)
					? recommendationDatesByInspectionId.get(rowInspectionId)
					: undefined;

				if (!recommendationDates || recommendationDates.size === 0) {
					return normalized;
				}

				const mergedRecommendationDates = Array.from(recommendationDates).sort(
					(left, right) => left.localeCompare(right, "pl", { sensitivity: "base" }),
				);

				return {
					...normalized,
					dataZalecen: mergedRecommendationDates.join(", "),
				};
			});

			const leaderMap: Record<string, number | null> = {};
			const relationMap: Record<string, number[]> = {};
			const acceptanceDatesMap: Record<string, string[]> = {};
			const noAcceptanceDatesMap: Record<
				string,
				InspectionNoAcceptanceDatesFlags
			> = {};
			const noLetterFlagsMap: Record<string, InspectionNoLetterFlags> = {};
			const canEditMap: Record<string, boolean> = {};
			const lockRecordIdsMap: Record<string, string[]> = {};
			const updatedAtMap: Record<string, string | null> = {};
			const shortValuesByRowId: Record<string, InspectionShortValuesByColumn> = {};
			const resolveBooleanFlag = (value: unknown) => {
				if (typeof value === "boolean") {
					return value;
				}

				if (typeof value === "number") {
					return value === 1;
				}

				if (typeof value === "string") {
					const normalized = value.trim().toLowerCase();
					return (
						normalized === "true" ||
						normalized === "1" ||
						normalized === "tak"
					);
				}

				return false;
			};
			const isLegacyNoLetterValue = (value: unknown) =>
				String(value ?? "").trim().toLowerCase() === "brak pisma";
			rawItems.forEach((rawRow, index) => {
				const normalizedRow = normalizeInspectionRow(
					(rawRow ?? {}) as RawInspectionRow,
					index,
				);
				const maybeLeaderId = (rawRow as { osobaKierujacaUserId?: unknown })
					.osobaKierujacaUserId;
				const maybeTeamIds = (rawRow as { teamMemberUserIds?: unknown })
					.teamMemberUserIds;
				const maybeAcceptanceDates = (
					rawRow as { dataAkceptacjiNotyList?: unknown }
				).dataAkceptacjiNotyList;
				const maybeCanEdit = (rawRow as { canEdit?: unknown }).canEdit;
				const noAcceptanceDates = resolveBooleanFlag(
					(rawRow as { brakDatAkceptacjiNoty?: unknown }).brakDatAkceptacjiNoty ??
						(rawRow as { brakDataAkceptacjiNotyList?: unknown })
							.brakDataAkceptacjiNotyList,
				);
				const noLetterDoreczenia = resolveBooleanFlag(
					(rawRow as { brakDataDoreczeniaPisma?: unknown })
						.brakDataDoreczeniaPisma,
				);
				const noLetterZastrzezenia = resolveBooleanFlag(
					(rawRow as { brakDataPismaZastrzezenia?: unknown })
						.brakDataPismaZastrzezenia,
				);
				const noLetterWyslaniaZastrzezen = resolveBooleanFlag(
					(rawRow as { brakDataWyslaniaPismaZZastrzezeniami?: unknown })
						.brakDataWyslaniaPismaZZastrzezeniami,
				);
				const noLetterWplywu = resolveBooleanFlag(
					(rawRow as { brakDataWplywuPisma?: unknown }).brakDataWplywuPisma,
				);
				const noLetterOdpowiedzi = resolveBooleanFlag(
					(rawRow as { brakDataPismaZOdpowiedzia?: unknown })
						.brakDataPismaZOdpowiedzia,
				);
				const noLetterWyslaniaOdpowiedzi = resolveBooleanFlag(
					(rawRow as { brakDataWyslaniaPismaZOdpowiedzia?: unknown })
						.brakDataWyslaniaPismaZOdpowiedzia,
				);
				const maybeUpdatedAt =
					(rawRow as { zaktualizowanoO?: unknown }).zaktualizowanoO ??
					(rawRow as { updatedAt?: unknown }).updatedAt;

				const legacyAcceptanceDate = toDateInputValue(
					normalizedRow.dataAkceptacjiNoty,
				);

				leaderMap[normalizedRow.id] =
					typeof maybeLeaderId === "number" && Number.isFinite(maybeLeaderId)
						? maybeLeaderId
						: null;
				relationMap[normalizedRow.id] = Array.isArray(maybeTeamIds)
					? maybeTeamIds.filter(
							(value): value is number =>
								typeof value === "number" && Number.isFinite(value),
						)
					: [];
				acceptanceDatesMap[normalizedRow.id] = toDateList(maybeAcceptanceDates)
					.length
					? toDateList(maybeAcceptanceDates)
					: legacyAcceptanceDate
						? [legacyAcceptanceDate]
						: [];
				noAcceptanceDatesMap[normalizedRow.id] = {
					brakDatAkceptacjiNoty: noAcceptanceDates,
				};
				noLetterFlagsMap[normalizedRow.id] = {
					brakDataDoreczeniaPisma:
						noLetterDoreczenia ||
						isLegacyNoLetterValue(
							(rawRow as { dataDoreczeniaPisma?: unknown }).dataDoreczeniaPisma,
						),
					brakDataPismaZastrzezenia:
						noLetterZastrzezenia ||
						isLegacyNoLetterValue(
							(rawRow as { dataPismaZastrzezenia?: unknown })
								.dataPismaZastrzezenia,
						),
					brakDataWyslaniaPismaZZastrzezeniami:
						noLetterWyslaniaZastrzezen ||
						isLegacyNoLetterValue(
							(rawRow as { dataWyslaniaPismaZZastrzezeniami?: unknown })
								.dataWyslaniaPismaZZastrzezeniami,
						),
					brakDataWplywuPisma:
						noLetterWplywu ||
						isLegacyNoLetterValue(
							(rawRow as { dataWplywuPisma?: unknown }).dataWplywuPisma,
						),
					brakDataPismaZOdpowiedzia:
						noLetterOdpowiedzi ||
						isLegacyNoLetterValue(
							(rawRow as { dataPismaZOdpowiedzia?: unknown })
								.dataPismaZOdpowiedzia,
						),
					brakDataWyslaniaPismaZOdpowiedzia:
						noLetterWyslaniaOdpowiedzi ||
						isLegacyNoLetterValue(
							(rawRow as { dataWyslaniaPismaZOdpowiedzia?: unknown })
								.dataWyslaniaPismaZOdpowiedzia,
						),
				};
				canEditMap[normalizedRow.id] =
					typeof maybeCanEdit === "boolean" ? maybeCanEdit : false;
				lockRecordIdsMap[normalizedRow.id] = resolveInspectionLockRecordIds(
					(rawRow ?? {}) as RawInspectionRow,
					normalizedRow.id,
				);
				updatedAtMap[normalizedRow.id] =
					typeof maybeUpdatedAt === "string" && maybeUpdatedAt.trim()
						? maybeUpdatedAt.trim()
						: null;
				shortValuesByRowId[normalizedRow.id] = {
					nazwaPodmiotu: String(
						(rawRow as { nazwaPodmiotuSkrocona?: unknown }).nazwaPodmiotuSkrocona ??
							(rawRow as { nazwaPodmiotuSkrot?: unknown }).nazwaPodmiotuSkrot ??
							"",
					).trim(),
					typInspekcji: String(
						(rawRow as { typInspekcjiSkrocona?: unknown }).typInspekcjiSkrocona ??
							(rawRow as { typInspekcjiSkrot?: unknown }).typInspekcjiSkrot ??
							"",
					).trim(),
					zakresInspekcji: String(
						(rawRow as { zakresInspekcjiSkrocona?: unknown })
							.zakresInspekcjiSkrocona ??
							(rawRow as { zakresInspekcjiSkrot?: unknown }).zakresInspekcjiSkrot ??
							"",
					).trim(),
					rodzajPodmiotu: String(
						(rawRow as { rodzajPodmiotuSkrocona?: unknown }).rodzajPodmiotuSkrocona ??
							(rawRow as { rodzajPodmiotuSkrot?: unknown }).rodzajPodmiotuSkrot ??
							"",
					).trim(),
					status: String(
						(rawRow as { statusSkrocona?: unknown }).statusSkrocona ??
							(rawRow as { statusSkrot?: unknown }).statusSkrot ??
							"",
					).trim(),
				};
			});

			setInspectionLeaderUserIdByRowId(leaderMap);
			setInspectionTeamMemberIdsByRowId(relationMap);
			setInspectionAcceptanceDatesByRowId(acceptanceDatesMap);
			setInspectionNoAcceptanceDatesByRowId(noAcceptanceDatesMap);
			setInspectionNoLetterFlagsByRowId(noLetterFlagsMap);
			setInspectionCanEditByRowId(canEditMap);
			setInspectionLockRecordIdsByRowId(lockRecordIdsMap);
			setInspectionUpdatedAtByRowId(updatedAtMap);
			setInspectionShortValuesByRowId(shortValuesByRowId);
			setInspectionRows(items);
			setSelectedInspectionId((prev) =>
				prev && items.some((row) => row.id === prev) ? prev : null,
			);
		} catch (error) {
			setRowsError(
				error instanceof Error && error.message
					? error.message
					: "Nie udało się pobrać danych Ewidencji kontroli z backendu.",
			);
			setInspectionRows([]);
			setSelectedInspectionId(null);
			setInspectionLeaderUserIdByRowId({});
			setInspectionTeamMemberIdsByRowId({});
			setInspectionAcceptanceDatesByRowId({});
			setInspectionNoAcceptanceDatesByRowId({});
			setInspectionNoLetterFlagsByRowId({});
			setInspectionCanEditByRowId({});
			setInspectionLockRecordIdsByRowId({});
			setInspectionUpdatedAtByRowId({});
			setInspectionShortValuesByRowId({});
		} finally {
			setIsRowsLoading(false);
		}
	}, [operatorLogin]);

	useEffect(() => {
		void loadInspections();
	}, [loadInspections]);

	useEffect(() => {
		if (typeof window === "undefined") {
			return;
		}

		const fromSession = window.sessionStorage.getItem(
			DASHBOARD_OPEN_INSPECTION_CODE_KEY,
		);
		if (fromSession?.trim()) {
			setPendingDashboardInspectionCode(fromSession.trim());
		}

		const handleOpenInspectionFromDashboard = (event: Event) => {
			const customEvent = event as CustomEvent<{ inspectionCode?: unknown }>;
			const inspectionCode =
				typeof customEvent.detail?.inspectionCode === "string"
					? customEvent.detail.inspectionCode.trim()
					: "";
			if (!inspectionCode) {
				return;
			}

			window.sessionStorage.setItem(
				DASHBOARD_OPEN_INSPECTION_CODE_KEY,
				inspectionCode,
			);
			setPendingDashboardInspectionCode(inspectionCode);
		};

		window.addEventListener(
			DASHBOARD_OPEN_INSPECTION_EVENT,
			handleOpenInspectionFromDashboard,
		);

		return () => {
			window.removeEventListener(
				DASHBOARD_OPEN_INSPECTION_EVENT,
				handleOpenInspectionFromDashboard,
			);
		};
	}, []);

	useEffect(() => {
		const handleRecommendationsChanged = () => {
			void loadInspections();
		};

		window.addEventListener(
			RECOMMENDATIONS_CHANGED_EVENT,
			handleRecommendationsChanged,
		);
		return () => {
			window.removeEventListener(
				RECOMMENDATIONS_CHANGED_EVENT,
				handleRecommendationsChanged,
			);
		};
	}, [loadInspections]);

	const loadInspectionDictionaries = useCallback(async () => {
		const resolveOptions = (
			result: Awaited<ReturnType<typeof fetchDictionaryEntries>>,
		) => {
			if (!result.ok) {
				return [];
			}

			return mapDictionaryEntriesToOptions(result.data);
		};

		const resolveSelectOptions = (
			result: Awaited<ReturnType<typeof fetchDictionaryEntries>>,
		) => {
			if (!result.ok) {
				return [];
			}

			return mapDictionaryEntriesToSelectOptions(result.data);
		};

		try {
			const normalizedOperatorLogin = operatorLogin.trim().toLowerCase();
			const [
				entityNamesResult,
				inspectionTypesResult,
				inspectionScopesResult,
				marketsResult,
				entityTypesResult,
				inspectionStatusesResult,
			] = await Promise.all([
				fetchDictionaryEntries("nazwy_podmiotow"),
				fetchDictionaryEntries("typy_inspekcji"),
				fetchDictionaryEntries("zakresy_inspekcji"),
				fetchDictionaryEntries("rynki"),
				fetchDictionaryEntries("rodzaje_podmiotu"),
				fetchDictionaryEntries("statusy_inspekcji"),
			]);

			let users: InspectionPeopleOption[] = [];
			const usersResponse = await fetch("/api/inspections/people-options", {
				method: "GET",
				headers: {
					"Content-Type": "application/json",
					"X-Operator-Login": operatorLogin,
				},
				cache: "no-store",
			});

			if (usersResponse.ok) {
				const payload = (await usersResponse.json()) as unknown;
				users = Array.isArray(payload)
					? payload
							.map((item) => {
								const raw = (item ?? {}) as {
									id?: unknown;
									login?: unknown;
									displayName?: unknown;
									active?: unknown;
									listVisibility?: unknown;
									widocznoscNaLiscie?: unknown;
									visibleOnList?: unknown;
									canBeLeader?: unknown;
									createdByOperator?: unknown;
									addedByOperator?: unknown;
									createdByLogin?: unknown;
									addedByLogin?: unknown;
									operatorLogin?: unknown;
									createdBy?: unknown;
									createdByUserLogin?: unknown;
									creatorLogin?: unknown;
									createdByOperatorLogin?: unknown;
									addedBy?: unknown;
									teamId?: unknown;
									zespolId?: unknown;
									teamName?: unknown;
								};

								const id =
									typeof raw.id === "number" && Number.isFinite(raw.id)
										? raw.id
										: 0;
								const login =
									typeof raw.login === "string" ? raw.login.trim() : "";
								if (!id || !login) {
									return null;
								}

								const displayName =
									typeof raw.displayName === "string"
										? raw.displayName.trim()
										: "";
								const active = raw.active !== false;
								const listVisibilityRaw =
									typeof raw.listVisibility === "string"
										? raw.listVisibility.trim().toLowerCase()
										: typeof raw.widocznoscNaLiscie === "string"
											? raw.widocznoscNaLiscie.trim().toLowerCase()
											: null;
								const isVisibleOnList =
									typeof raw.visibleOnList === "boolean"
										? raw.visibleOnList
										: typeof raw.visibleOnList === "string"
											? raw.visibleOnList.trim().toLowerCase() === "true"
											: typeof raw.visibleOnList === "number"
												? raw.visibleOnList === 1
										: listVisibilityRaw === "hidden" ||
											  listVisibilityRaw === "ukryty"
											? false
											: listVisibilityRaw === "visible" ||
											    listVisibilityRaw === "widoczny"
										  ? true
										  : active;
								const numericTeamId = Number(raw.teamId ?? raw.zespolId);
								const teamId =
									Number.isFinite(numericTeamId) && numericTeamId > 0
										? numericTeamId
										: null;
								const teamName =
									typeof raw.teamName === "string" ? raw.teamName : null;
								const createdByObjectLogin =
									raw.createdBy && typeof raw.createdBy === "object"
										? ((raw.createdBy as { login?: unknown }).login ?? "")
										: "";
								const addedByObjectLogin =
									raw.addedBy && typeof raw.addedBy === "object"
										? ((raw.addedBy as { login?: unknown }).login ?? "")
										: "";
								const creatorLoginRaw =
									typeof raw.createdByLogin === "string"
										? raw.createdByLogin
										: typeof raw.addedByLogin === "string"
											? raw.addedByLogin
											: typeof raw.createdByUserLogin === "string"
												? raw.createdByUserLogin
												: typeof raw.creatorLogin === "string"
													? raw.creatorLogin
													: typeof raw.createdByOperatorLogin === "string"
														? raw.createdByOperatorLogin
														: typeof raw.operatorLogin === "string"
															? raw.operatorLogin
															: typeof raw.createdBy === "string"
																? raw.createdBy
																: typeof createdByObjectLogin === "string"
																	? createdByObjectLogin
																	: typeof addedByObjectLogin === "string"
																		? addedByObjectLogin
																		: "";
								const normalizedCreatorLogin = creatorLoginRaw
									.trim()
									.toLowerCase();
								const normalizeBooleanLike = (value: unknown) => {
									if (typeof value === "boolean") {
										return value;
									}

									if (typeof value === "number") {
										return value === 1;
									}

									if (typeof value === "string") {
										const normalized = value.trim().toLowerCase();
										return (
											normalized === "true" ||
											normalized === "1" ||
											normalized === "tak"
										);
									}

									return false;
								};
								const createdByOperator =
									normalizeBooleanLike(raw.createdByOperator) ||
									normalizeBooleanLike(raw.addedByOperator) ||
									(Boolean(normalizedCreatorLogin) &&
										normalizedCreatorLogin === normalizedOperatorLogin);

								return {
									id,
									login,
									displayName: displayName || login,
									active,
									visibleOnList: isVisibleOnList,
									canBeLeader:
										typeof raw.canBeLeader === "boolean"
											? raw.canBeLeader
											: typeof raw.canBeLeader === "string"
											? ["true", "1", "tak"].includes(
													raw.canBeLeader.trim().toLowerCase(),
												)
											: typeof raw.canBeLeader === "number"
												? raw.canBeLeader === 1
											: false,
									createdByOperator,
									teamId,
									teamName,
								};
							})
							.filter((user): user is InspectionPeopleOption => user !== null)
					: [];
			}

			const activeUsers = users.filter((user) => user.visibleOnList);
			setAllUsers(users);
			setActiveUsers(activeUsers);

			const operatorUser = users.find(
				(user) => user.login.trim().toLowerCase() === normalizedOperatorLogin,
			);
			setOperatorDisplayName(
				operatorUser ? getUserDisplayName(operatorUser) : operatorLogin.trim(),
			);
			setOperatorUserId(operatorUser?.id ?? null);
			setOperatorTeamId(operatorUser?.teamId ?? null);

			setEntityNameOptions(resolveSelectOptions(entityNamesResult));
			setInspectionTypeOptions(resolveOptions(inspectionTypesResult));
			const inspectionScopeSelectOptions = resolveSelectOptions(
				inspectionScopesResult,
			);
			const inspectionScopeMapByValue: Record<string, string> = {};
			if (inspectionScopesResult.ok) {
				for (const entry of inspectionScopesResult.data) {
					const scopeValue = entry.nazwaPozycji.trim();
					const userLabel = (entry.nazwaUzytkowa ?? "").trim();

					if (!scopeValue || !userLabel || inspectionScopeMapByValue[scopeValue]) {
						continue;
					}

					inspectionScopeMapByValue[scopeValue] = userLabel;
				}
			}
			setInspectionScopeMapByValue(inspectionScopeMapByValue);
			setInspectionScopeOptions(
				normalizeInspectionScopeValues(
					inspectionScopeSelectOptions.map((option) => option.value),
				).map((value) => {
					return { value, label: value };
				}),
			);
			setMarketOptions(resolveOptions(marketsResult));
			setEntityTypeOptions(resolveOptions(entityTypesResult));
			setInspectionStatusOptions(
				resolveSelectOptions(inspectionStatusesResult).map((option) => ({
					value: option.value,
					label: option.value,
				})),
			);
		} catch {
			setEntityNameOptions([]);
			setInspectionTypeOptions([]);
			setInspectionScopeOptions([]);
			setInspectionScopeMapByValue({});
			setMarketOptions([]);
			setEntityTypeOptions([]);
			setInspectionStatusOptions([]);
			setAllUsers([]);
			setActiveUsers([]);
			setOperatorUserId(null);
			setOperatorTeamId(null);
			setSelectedLeaderUserId(null);
			setOperatorDisplayName(operatorLogin.trim());
		}
	}, [operatorLogin]);

	useEffect(() => {
		void loadInspectionDictionaries();
	}, [loadInspectionDictionaries]);

	useEffect(() => {
		if (typeof window === "undefined") {
			return;
		}

		const raw = window.localStorage.getItem(columnWidthsStorageKey);
		if (!raw) {
			setColumnWidths(DEFAULT_INSPECTIONS_COLUMN_WIDTHS);
			setAreColumnWidthsHydrated(true);
			return;
		}

		try {
			const parsed = JSON.parse(raw) as Record<string, unknown>;
			const next: Partial<Record<InspectionColumnKey, number>> = {};
			for (const [key, value] of Object.entries(parsed)) {
				const width = Number(value);
				if (!Number.isFinite(width)) {
					continue;
				}

				const columnKey = key as InspectionColumnKey;
				const normalizedWidth = Math.max(
					INSPECTIONS_MIN_COLUMN_WIDTH,
					Math.min(1200, Math.round(width)),
				);
				next[columnKey] = normalizedWidth;
			}

			setColumnWidths({
				...DEFAULT_INSPECTIONS_COLUMN_WIDTHS,
				...next,
			});
		} catch {
			setColumnWidths(DEFAULT_INSPECTIONS_COLUMN_WIDTHS);
		}

		setAreColumnWidthsHydrated(true);
	}, [columnWidthsStorageKey]);

	useEffect(() => {
		if (typeof window === "undefined") {
			return;
		}

		const raw = window.localStorage.getItem(nameVariantsStorageKey);
		if (!raw) {
			setAreNameVariantsHydrated(true);
			return;
		}

		try {
			const parsed = JSON.parse(raw) as Partial<Record<InspectionColumnKey, unknown>>;
			const next: InspectionNameVariantByColumn = {
				...DEFAULT_INSPECTION_NAME_VARIANTS,
			};

			for (const columnKey of INSPECTION_NAME_VARIANT_COLUMN_KEYS) {
				const value = parsed[columnKey];
				if (
					isInspectionNameVariant(value) &&
					isInspectionNameVariantAllowedForColumn(columnKey, value)
				) {
					next[columnKey] = value;
				}
			}

			setInspectionNameVariants(next);
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

		window.localStorage.setItem(
			nameVariantsStorageKey,
			JSON.stringify(inspectionNameVariants),
		);
	}, [areNameVariantsHydrated, inspectionNameVariants, nameVariantsStorageKey]);

	const hasCustomColumnWidths = useMemo(() => {
		const keys = new Set<string>([
			...Object.keys(DEFAULT_INSPECTIONS_COLUMN_WIDTHS),
			...Object.keys(columnWidths),
		]);

		for (const key of keys) {
			const columnKey = key as InspectionColumnKey;
			const currentWidth = columnWidths[columnKey];
			const defaultWidth = DEFAULT_INSPECTIONS_COLUMN_WIDTHS[columnKey];

			if (typeof currentWidth === "number") {
				if (typeof defaultWidth !== "number" || currentWidth !== defaultWidth) {
					return true;
				}
				continue;
			}

			if (typeof defaultWidth === "number") {
				return true;
			}
		}

		return false;
	}, [columnWidths]);

	useEffect(() => {
		if (typeof window === "undefined") {
			return;
		}

		if (!areColumnWidthsHydrated) {
			return;
		}

		if (!hasCustomColumnWidths) {
			window.localStorage.removeItem(columnWidthsStorageKey);
			return;
		}

		window.localStorage.setItem(
			columnWidthsStorageKey,
			JSON.stringify(columnWidths),
		);
	}, [
		areColumnWidthsHydrated,
		columnWidths,
		columnWidthsStorageKey,
		hasCustomColumnWidths,
	]);

	const handleResizeColumn = useCallback(
		(columnKey: InspectionColumnKey, width: number) => {
			setColumnWidths((prev) => ({
				...prev,
				[columnKey]: Math.max(
					INSPECTIONS_MIN_COLUMN_WIDTH,
					Math.min(1200, Math.round(width)),
				),
			}));
		},
		[],
	);

	const handleResetColumnWidths = useCallback(() => {
		setColumnWidths(DEFAULT_INSPECTIONS_COLUMN_WIDTHS);
		if (typeof window !== "undefined") {
			window.localStorage.removeItem(columnWidthsStorageKey);
		}
	}, [columnWidthsStorageKey]);

	const loadInspectionPeopleOptionsForEdit = useCallback(
		async (inspectionId: string) => {
			try {
				const query = new URLSearchParams();
				const normalizedInspectionId = inspectionId.trim();
				if (normalizedInspectionId) {
					query.set("inspectionId", normalizedInspectionId);
				}
				const querySuffix = query.toString() ? `?${query.toString()}` : "";

				const response = await fetch(
					`/api/inspections/people-options${querySuffix}`,
					{
						method: "GET",
						headers: {
							"Content-Type": "application/json",
							"X-Operator-Login": operatorLogin,
						},
						cache: "no-store",
					},
				);

				if (!response.ok) {
					return;
				}

				const payload = (await response.json()) as unknown;
				const normalizedOperatorLogin = operatorLogin.trim().toLowerCase();
				const users = Array.isArray(payload)
					? payload
							.map((item) => {
								const raw = (item ?? {}) as {
									id?: unknown;
									login?: unknown;
									displayName?: unknown;
									active?: unknown;
									visibleOnList?: unknown;
									listVisibility?: unknown;
									widocznoscNaLiscie?: unknown;
									canBeLeader?: unknown;
									teamId?: unknown;
									zespolId?: unknown;
									teamName?: unknown;
									createdByOperator?: unknown;
									addedByOperator?: unknown;
									createdByLogin?: unknown;
									addedByLogin?: unknown;
									createdByOperatorLogin?: unknown;
									creatorLogin?: unknown;
									operatorLogin?: unknown;
								};

								const id =
									typeof raw.id === "number" && Number.isFinite(raw.id)
										? raw.id
										: 0;
								const login =
									typeof raw.login === "string" ? raw.login.trim() : "";
								if (!id || !login) {
									return null;
								}

								const normalizeBooleanLike = (value: unknown) => {
									if (typeof value === "boolean") {
										return value;
									}
									if (typeof value === "number") {
										return value === 1;
									}
									if (typeof value === "string") {
										const normalized = value.trim().toLowerCase();
										return (
											normalized === "true" ||
											normalized === "1" ||
											normalized === "tak"
										);
									}
									return false;
								};

								const creatorLoginRaw =
									typeof raw.createdByLogin === "string"
										? raw.createdByLogin
										: typeof raw.addedByLogin === "string"
											? raw.addedByLogin
											: typeof raw.createdByOperatorLogin === "string"
												? raw.createdByOperatorLogin
												: typeof raw.creatorLogin === "string"
													? raw.creatorLogin
													: typeof raw.operatorLogin === "string"
														? raw.operatorLogin
														: "";

								const listVisibilityRaw =
									typeof raw.listVisibility === "string"
										? raw.listVisibility.trim().toLowerCase()
										: typeof raw.widocznoscNaLiscie === "string"
											? raw.widocznoscNaLiscie.trim().toLowerCase()
											: "";
								const active = raw.active !== false;
								const visibleOnList =
									typeof raw.visibleOnList === "boolean"
										? raw.visibleOnList
										: listVisibilityRaw === "hidden" || listVisibilityRaw === "ukryty"
											? false
											: active;

								const numericTeamId = Number(raw.teamId ?? raw.zespolId);

								return {
									id,
									login,
									displayName:
										typeof raw.displayName === "string" && raw.displayName.trim()
											? raw.displayName.trim()
											: login,
									active,
									visibleOnList,
									canBeLeader: normalizeBooleanLike(raw.canBeLeader),
									createdByOperator:
										normalizeBooleanLike(raw.createdByOperator) ||
										normalizeBooleanLike(raw.addedByOperator) ||
										(Boolean(creatorLoginRaw) &&
											creatorLoginRaw.trim().toLowerCase() === normalizedOperatorLogin),
									teamId:
										Number.isFinite(numericTeamId) && numericTeamId > 0
											? numericTeamId
											: null,
									teamName:
										typeof raw.teamName === "string" ? raw.teamName : null,
								};
							})
							.filter((user): user is InspectionPeopleOption => user !== null)
					: [];

				const visibleUsers = users.filter((user) => user.visibleOnList);
				setAllUsers(users);
				setActiveUsers(visibleUsers);
				const operatorUser = users.find(
					(user) => user.login.trim().toLowerCase() === normalizedOperatorLogin,
				);
				setOperatorDisplayName(
					operatorUser ? getUserDisplayName(operatorUser) : operatorLogin.trim(),
				);
				setOperatorUserId(operatorUser?.id ?? null);
				setOperatorTeamId(operatorUser?.teamId ?? null);
			} catch {
				// Keep existing people options when scoped refresh fails.
			}
		},
		[operatorLogin],
	);

	const inspectionRowsForDisplay = useMemo(
		() => {
			const inspectionScopeMap = new Map<string, string>();
			for (const [scopeValue, userLabel] of Object.entries(
				inspectionScopeMapByValue,
			)) {
				const normalizedScopeValue = scopeValue.trim().toLowerCase();
				const normalizedUserLabel = userLabel.trim();
				if (!normalizedScopeValue || !normalizedUserLabel) {
					continue;
				}

				inspectionScopeMap.set(normalizedScopeValue, normalizedUserLabel);
			}

			return inspectionRows.map((row) => {
				const shortValues = inspectionShortValuesByRowId[row.id];
				if (!shortValues) {
					return row;
				}

				const getDisplayValue = (columnKey: InspectionNameVariantColumnKey) => {
					const shortValue = shortValues[columnKey]?.trim() ?? "";

					if (columnKey === "zakresInspekcji") {
						if (inspectionNameVariants.zakresInspekcji === "short" && shortValue) {
							return shortValue;
						}

						if (inspectionNameVariants.zakresInspekcji === "user") {
							const scopes = parseMultiValueField(row.zakresInspekcji);
							if (scopes.length === 0) {
								return "";
							}

							const mappedScopes = scopes.map((scope) => {
								const mapped = inspectionScopeMap.get(scope.trim().toLowerCase());
								return mapped ?? "";
							});

							return joinMultiValueField(mappedScopes);
						}
					}

					return inspectionNameVariants[columnKey] === "short" && shortValue
						? shortValue
						: row[columnKey];
				};

				return {
					...row,
					nazwaPodmiotu: getDisplayValue("nazwaPodmiotu"),
					typInspekcji: getDisplayValue("typInspekcji"),
					zakresInspekcji: getDisplayValue("zakresInspekcji"),
					rodzajPodmiotu: getDisplayValue("rodzajPodmiotu"),
					status: getDisplayValue("status"),
				};
			});
		},
		[
			inspectionNameVariants,
			inspectionRows,
			inspectionScopeMapByValue,
			inspectionShortValuesByRowId,
		],
	);

	const {
		advancedFilterAnchor,
		advancedFilterColumnKey,
		advancedFilterSearch,
		advancedFilters,
		clearAdvancedFilterForSelectedColumn,
		clearFilters,
		canClearFilters,
		columnFilters,
		draftHiddenColumns,
		draftSelectableColumnDefinitions,
		draftVisibleInspectionColumnsCount,
		draftSelectedInspectionView,
		filteredAndSortedInspectionRows,
		paginatedInspectionRows,
		currentPage,
		totalPages,
		paginationItems,
		handlePageChange,
		handlePageSizeChange,
		pageSize,
		handleApplyViewChanges,
		handleDraftColumnVisibilityChange,
		handleDraftDeselectAllColumns,
		handleDraftResetSelection,
		handleDraftSelectAllColumns,
		handleDraftViewSelect,
		handleFilterChange,
		handleOpenViewModal,
		handleSortByColumn,
		isAdvancedFilterModalOpen,
		isColumnPickerOpen,
		openAdvancedFilterForColumn,
		selectedAdvancedFilterDateRange,
		selectedAdvancedFilterValues,
		selectedInspectionView,
		selectAllVisibleAdvancedFilterValues,
		setAdvancedFilterSearch,
		setAdvancedFilterDateRange,
		setDraftHiddenColumns,
		setIsAdvancedFilterModalOpen,
		setIsColumnPickerOpen,
		sortColumnKey,
		sortDirection,
		toggleAdvancedFilterValue,
		visibleAdvancedFilterValues,
		visibleInspectionColumnDefinitions,
	} = useInspectionsTableState({
		inspectionRows: inspectionRowsForDisplay,
		tableViewStorageKey,
		tableViewStorageArea: "localStorage",
	});

	const columnDisplayModeOptionsByKey = useMemo(
		() =>
			Object.fromEntries(
				INSPECTION_NAME_VARIANT_COLUMN_KEYS.map((columnKey) => [
					columnKey,
					[...INSPECTION_NAME_VARIANT_OPTIONS],
				]),
			) as Partial<
				Record<
					InspectionColumnKey,
					Array<{ value: string; label: string }>
				>
			>,
		[],
	);

	const draftColumnDisplayModeValuesByKey = useMemo(
		() =>
			Object.fromEntries(
				INSPECTION_NAME_VARIANT_COLUMN_KEYS.map((columnKey) => [
					columnKey,
					draftInspectionNameVariants[columnKey],
				]),
			) as Partial<Record<InspectionColumnKey, string>>,
		[draftInspectionNameVariants],
	);

	const handleOpenInspectionViewModal = () => {
		setDraftInspectionNameVariants(inspectionNameVariants);
		handleOpenViewModal();
	};

	const handleApplyInspectionViewChanges = () => {
		setInspectionNameVariants(draftInspectionNameVariants);
		handleApplyViewChanges();
	};

	const handleResetInspectionViewSelection = () => {
		handleDraftResetSelection();
		setDraftInspectionNameVariants(DEFAULT_INSPECTION_NAME_VARIANTS);
	};

	useEffect(() => {
		if (!pendingDashboardInspectionCode || isRowsLoading) {
			return;
		}

		const normalizedCode = pendingDashboardInspectionCode.trim().toLowerCase();
		if (!normalizedCode) {
			setPendingDashboardInspectionCode(null);
			return;
		}

		const targetRow = filteredAndSortedInspectionRows.find(
			(row) => row.kodInspekcji.trim().toLowerCase() === normalizedCode,
		);

		if (!targetRow) {
			return;
		}

		const rowIndex = filteredAndSortedInspectionRows.findIndex(
			(row) => row.id === targetRow.id,
		);
		if (rowIndex < 0) {
			return;
		}

		const targetPage = Math.floor(rowIndex / pageSize) + 1;
		handlePageChange(targetPage);
		setSelectedInspectionId(targetRow.id);
		setFlashInspectionId(targetRow.id);
		setPendingDashboardInspectionCode(null);

		if (typeof window !== "undefined") {
			window.sessionStorage.removeItem(DASHBOARD_OPEN_INSPECTION_CODE_KEY);
		}

		window.setTimeout(() => {
			setFlashInspectionId((current) =>
				current === targetRow.id ? null : current,
			);
		}, 2200);
	}, [
		filteredAndSortedInspectionRows,
		handlePageChange,
		isRowsLoading,
		pageSize,
		pendingDashboardInspectionCode,
	]);

	useEffect(() => {
		if (!selectedInspectionId) {
			return;
		}

		const isSelectedVisibleOnPage = paginatedInspectionRows.some(
			(row) => row.id === selectedInspectionId,
		);

		if (!isSelectedVisibleOnPage) {
			setSelectedInspectionId(null);
		}
	}, [paginatedInspectionRows, selectedInspectionId]);

	const selectedTeamMembers = useMemo(() => {
		return selectedTeamMemberIds
			.map((userId) => {
				const user = allUsers.find((item) => item.id === userId);
				return user ? getUserDisplayName(user) : null;
			})
			.filter((name): name is string => Boolean(name));
	}, [allUsers, selectedTeamMemberIds]);

	useEffect(() => {
		if (!teamMemberScopeError && outOfScopeTeamMemberUserId === null) {
			return;
		}

		setTeamMemberScopeError(null);
		setOutOfScopeTeamMemberUserId(null);
	}, [
		selectedTeamMemberIds,
		teamMemberScopeError,
		outOfScopeTeamMemberUserId,
	]);

	const availableLeaderUsers = useMemo(() => {
		const sourceUsers = allUsers.filter((user) => user.canBeLeader);

		if (!operatorUserId) {
			return sourceUsers;
		}

		if (authRole === "director") {
			return sourceUsers;
		}

		if (authRole === "team_lead") {
			return sourceUsers.filter((user) => {
				if (user.id === operatorUserId) {
					return true;
				}

				if (operatorTeamId !== null && user.teamId === operatorTeamId) {
					return true;
				}

				return user.createdByOperator;
			});
		}

		if (authRole === "inspector") {
			return sourceUsers.filter((user) => user.id === operatorUserId);
		}

		return sourceUsers.filter((user) => user.id === operatorUserId);
	}, [allUsers, authRole, operatorTeamId, operatorUserId]);


	const leaderOptionsForModal = useMemo(() => {
		if (!isEditMode || selectedLeaderUserId === null) {
			return availableLeaderUsers;
		}

		if (availableLeaderUsers.some((user) => user.id === selectedLeaderUserId)) {
			return availableLeaderUsers;
		}

		const currentLeader = allUsers.find((user) => user.id === selectedLeaderUserId);
		if (!currentLeader) {
			return availableLeaderUsers;
		}

		return [...availableLeaderUsers, currentLeader];
	}, [
		allUsers,
		availableLeaderUsers,
		isEditMode,
		selectedLeaderUserId,
	]);

	const leaderChangeIrreversibleWarning = useMemo(() => {
		if (authRole !== "team_lead") {
			return null;
		}

		if (!isEditMode || currentEditingInspectionLeaderUserId === null) {
			return null;
		}

		const isCurrentLeaderAvailable = availableLeaderUsers.some(
			(user) => user.id === currentEditingInspectionLeaderUserId,
		);
		if (isCurrentLeaderAvailable) {
			return null;
		}

		return "Osoba kierująca jest spoza Twojego zespołu. Jeśli ją zmienisz i zapiszesz rekord, nie będziesz mógł wybrać jej ponownie.";
	}, [
		authRole,
		availableLeaderUsers,
		currentEditingInspectionLeaderUserId,
		isEditMode,
	]);

	useEffect(() => {
		if (isEditMode) {
			return;
		}

		if (availableLeaderUsers.length === 0) {
			setSelectedLeaderUserId(null);
			return;
		}

		const isSelectedLeaderValid =
			selectedLeaderUserId !== null &&
			availableLeaderUsers.some((user) => user.id === selectedLeaderUserId);

		if (!isSelectedLeaderValid) {
			const defaultLeaderUserId =
				operatorUserId &&
				availableLeaderUsers.some((user) => user.id === operatorUserId)
					? operatorUserId
					: (availableLeaderUsers[0]?.id ?? null);
			setSelectedLeaderUserId(defaultLeaderUserId);
		}
	}, [availableLeaderUsers, isEditMode, operatorUserId, selectedLeaderUserId]);

	useEffect(() => {
		setAddInspectionForm((prev) => {
			if (!selectedLeaderUserId) {
				if (isEditMode) {
					return prev;
				}

				const fallback = operatorDisplayName.trim() || operatorLogin.trim();
				return prev.osobaKierujaca === fallback
					? prev
					: { ...prev, osobaKierujaca: fallback };
			}

			const selectedLeader =
				availableLeaderUsers.find((user) => user.id === selectedLeaderUserId) ??
				allUsers.find((user) => user.id === selectedLeaderUserId);
			const nextLeaderName = selectedLeader
				? getUserDisplayName(selectedLeader)
				: operatorDisplayName.trim() || operatorLogin.trim();
			return prev.osobaKierujaca === nextLeaderName
				? prev
				: { ...prev, osobaKierujaca: nextLeaderName };
		});
	}, [
		allUsers,
		availableLeaderUsers,
		isEditMode,
		operatorDisplayName,
		operatorLogin,
		selectedLeaderUserId,
	]);

	useEffect(() => {
		setAddInspectionForm((prev) => ({
			...prev,
			skladZespolu: selectedTeamMembers.join("; "),
		}));
	}, [selectedTeamMembers]);

	useEffect(() => {
		setAddInspectionForm((prev) => ({
			...prev,
			zakresInspekcji: joinMultiValueField(selectedInspectionScopes),
		}));
	}, [selectedInspectionScopes]);

	const handleExportCurrentView = useCallback(
		async (
			recommendationColumnKeys: RecommendationExportColumnKey[],
			sanctionColumnKeys: SanctionExportColumnKey[],
			decisionColumnKeys: DecisionExportColumnKey[],
			includeRecommendations: boolean,
			includeSanctions: boolean,
			includeDecisions: boolean,
		) => {
			if (
				isExporting ||
				filteredAndSortedInspectionRows.length === 0 ||
				visibleInspectionColumnDefinitions.length === 0
			) {
				return;
			}

			setIsExporting(true);
			setAddInspectionError(null);

			try {
				const workbook = await createStyledExportWorkbook("Ewidencja kontroli");

				const loadInspectionCodeMap = async (url: string) => {
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
							return new Map<number, string>();
						}

						const payload = (await response.json()) as
							| Array<{
									id?: unknown;
									lp?: unknown;
									inspectionLp?: unknown;
									inspectionKod?: unknown;
									kodInspekcji?: unknown;
							  }>
							| {
									items?: Array<{
										id?: unknown;
										lp?: unknown;
										inspectionLp?: unknown;
										inspectionKod?: unknown;
										kodInspekcji?: unknown;
									}>;
							  };
						const rawItems = Array.isArray(payload)
							? payload
							: (payload.items ?? []);

						return new Map(
							rawItems
								.map((item) => {
									const id = Number(item.id);
									if (!Number.isFinite(id) || id <= 0) {
										return null;
									}

									const inspectionCode = String(
										item.inspectionKod ??
											item.kodInspekcji ??
											item.inspectionLp ??
											item.lp ??
											"",
									).trim();
									if (!inspectionCode) {
										return null;
									}

									return [id, inspectionCode] as const;
								})
								.filter(
									(entry): entry is readonly [number, string] => entry !== null,
								),
						);
					} catch {
						return new Map<number, string>();
					}
				};

				const exportedInspectionIds = new Set(
					filteredAndSortedInspectionRows
						.map((row) => Number(row.id))
						.filter((id) => Number.isFinite(id) && id > 0),
				);

				const inspectionCodeById = new Map(
					filteredAndSortedInspectionRows
						.map((row) => {
							const numericId = Number(row.id);
							if (!Number.isFinite(numericId) || numericId <= 0) {
								return null;
							}

							return [numericId, row.kodInspekcji] as const;
						})
						.filter(
							(entry): entry is readonly [number, string] => entry !== null,
						),
				);

				const [
					recommendationsResult,
					sanctionRequestsResult,
					decisionsResult,
					recommendationCodeById,
					sanctionCodeById,
				] = await Promise.all([
					fetchRecommendations(operatorLogin, {
						sortBy: "id",
						sortOrder: "asc",
					}),
					fetchSanctionRequests(operatorLogin, {
						sortBy: "id",
						sortOrder: "asc",
					}),
					fetchObligatingDecisions(operatorLogin),
					loadInspectionCodeMap(RECOMMENDATIONS_AVAILABLE_INSPECTIONS_API_URL),
					loadInspectionCodeMap(SANCTIONS_AVAILABLE_INSPECTIONS_API_URL),
				]);

				const relatedRecommendationsSource = recommendationsResult.ok
					? recommendationsResult.data.items
					: [];
				const relatedSanctionRequestsSource = sanctionRequestsResult.ok
					? sanctionRequestsResult.data.items
					: [];
				const decisionsSource = decisionsResult.ok
					? decisionsResult.data.items
					: [];

				const relatedRecommendations = relatedRecommendationsSource.filter(
					(item) =>
						typeof item.inspectionId === "number" &&
						exportedInspectionIds.has(item.inspectionId),
				);

				const relatedSanctionRequests = relatedSanctionRequestsSource.filter(
					(item) =>
						typeof item.inspectionId === "number" &&
						exportedInspectionIds.has(item.inspectionId),
				);

				const recommendationInspectionIdByCode = new Map<string, number>();
				for (const recommendation of relatedRecommendationsSource) {
					const code = String(recommendation.kodZalecenia ?? "")
						.trim()
						.toUpperCase();
					if (!code) {
						continue;
					}

					if (
						typeof recommendation.inspectionId === "number" &&
						Number.isFinite(recommendation.inspectionId)
					) {
						recommendationInspectionIdByCode.set(
							code,
							recommendation.inspectionId,
						);
					}
				}

				const relatedDecisions = decisionsSource.filter((decision) => {
					const recommendationCode = String(
						decision.recommendationKodZalecenia ?? "",
					)
						.trim()
						.toUpperCase();
					if (!recommendationCode) {
						return false;
					}

					const relatedInspectionId =
						recommendationInspectionIdByCode.get(recommendationCode);
					return (
						typeof relatedInspectionId === "number" &&
						exportedInspectionIds.has(relatedInspectionId)
					);
				});

				const normalizeExportValue = (value: unknown) => {
					const normalized = String(value ?? "").trim();
					if (!normalized) {
						return "";
					}

					return normalized.toLowerCase() === "brak" ? "-" : normalized;
				};

				const isNotApplicableByInspectionType = (
					inspectionType: string,
					columnKey: InspectionColumnKey,
				) => {
					const normalizedType = inspectionType.trim().toLowerCase();
					const isControlType =
						normalizedType.includes("kontrol") ||
						normalizedType.startsWith("kont") ||
						normalizedType === "k";
					const isSupervisoryVisitType =
						normalizedType.includes("wizyta") ||
						normalizedType.startsWith("wiz") ||
						normalizedType === "w";

					if (isControlType && !isSupervisoryVisitType) {
						return (
							columnKey === "dataAkceptacjiSprawozdania" ||
							columnKey === "dataDoreczeniaPisma"
						);
					}

					if (isSupervisoryVisitType && !isControlType) {
						return (
							columnKey === "dataDoreczeniaProtokolu" ||
							columnKey === "dataWyslaniaPismaZOdpowiedzia" ||
							columnKey === "dataPismaZOdpowiedzia"
						);
					}

					return false;
				};

				const inspectionHeaders = visibleInspectionColumnDefinitions.map(
					(column) => column.label,
				);
				const inspectionRowsForExport = filteredAndSortedInspectionRows.map(
					(row) => {
						const noLetterFlags = inspectionNoLetterFlagsByRowId[row.id];
						const noAcceptanceDatesFlags = inspectionNoAcceptanceDatesByRowId[row.id];

						return visibleInspectionColumnDefinitions.map((column) => {
							const shouldExportNotApplicable = isNotApplicableByInspectionType(
								String(row.typInspekcji ?? ""),
								column.key,
							);
							const shouldExportNoLetter =
								(column.key === "dataDoreczeniaPisma" &&
									noLetterFlags?.brakDataDoreczeniaPisma) ||
								(column.key === "dataPismaZastrzezenia" &&
									noLetterFlags?.brakDataPismaZastrzezenia) ||
								(column.key === "dataWyslaniaPismaZZastrzezeniami" &&
									noLetterFlags?.brakDataWyslaniaPismaZZastrzezeniami) ||
								(column.key === "dataWplywuPisma" &&
									noLetterFlags?.brakDataWplywuPisma) ||
								(column.key === "dataPismaZOdpowiedzia" &&
									noLetterFlags?.brakDataPismaZOdpowiedzia) ||
								(column.key === "dataWyslaniaPismaZOdpowiedzia" &&
									noLetterFlags?.brakDataWyslaniaPismaZOdpowiedzia);
							const shouldExportNoAcceptanceDates =
								column.key === "dataAkceptacjiNoty" &&
								noAcceptanceDatesFlags?.brakDatAkceptacjiNoty;

							if (shouldExportNotApplicable) {
								return "Nie dotyczy";
							}

							if (shouldExportNoLetter) {
								return "Brak pisma";
							}

							if (shouldExportNoAcceptanceDates) {
								return "Brak pisma";
							}

							return normalizeExportValue(row[column.key]);
						});
					},
				);

				const getRecommendationExportValue = (
					item: (typeof relatedRecommendations)[number],
					key: RecommendationExportColumnKey,
				) => {
					const inspectionId = item.inspectionId ?? null;
					const inspectionCode =
						String(item.inspectionKod ?? "").trim() ||
						String(item.kodInspekcji ?? "").trim() ||
						String(item.inspectionLp ?? "").trim() ||
						(typeof inspectionId === "number"
							? (recommendationCodeById.get(inspectionId) ??
								inspectionCodeById.get(inspectionId) ??
								"")
							: "");

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
						case "dataZalecen":
							return item.dataZalecen ?? "";
						case "terminyWykonaniaZalecenList":
							return item.terminyWykonaniaZalecenList.join(", ");
						case "dataAkceptacjiNotyWeryfikacjiList":
							return item.dataAkceptacjiNotyWeryfikacjiList.join(", ");
						case "status":
							return item.status ?? "";
						case "komentarz":
							return item.komentarz ?? "";
					}
				};

				const recommendationHeaders = recommendationColumnKeys.map(
					(key) =>
						RECOMMENDATION_EXPORT_COLUMNS.find((column) => column.key === key)
							?.label ?? key,
				);
				const recommendationRowsForExport = relatedRecommendations.map((item) =>
					recommendationColumnKeys.map((key) =>
						normalizeExportValue(getRecommendationExportValue(item, key)),
					),
				);

				const getSanctionExportValue = (
					item: (typeof relatedSanctionRequests)[number],
					key: SanctionExportColumnKey,
				) => {
					const inspectionId = item.inspectionId ?? null;
					const inspectionCode =
						String(item.inspectionKod ?? "").trim() ||
						String(item.kodInspekcji ?? "").trim() ||
						String(item.inspectionLp ?? "").trim() ||
						(typeof inspectionId === "number"
							? (sanctionCodeById.get(inspectionId) ??
								inspectionCodeById.get(inspectionId) ??
								"")
							: "");

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
				};

				const sanctionHeaders = sanctionColumnKeys.map(
					(key) =>
						SANCTION_EXPORT_COLUMNS.find((column) => column.key === key)
							?.label ?? key,
				);
				const sanctionRowsForExport = relatedSanctionRequests.map((item) =>
					sanctionColumnKeys.map((key) =>
						normalizeExportValue(getSanctionExportValue(item, key)),
					),
				);

				const getDecisionExportValue = (
					item: (typeof relatedDecisions)[number],
					key: DecisionExportColumnKey,
					rowIndex: number,
				) => {
					const recommendationCode = String(
						item.recommendationKodZalecenia ?? "",
					).trim();
					const mappedInspectionId = recommendationCode
						? recommendationInspectionIdByCode.get(
								recommendationCode.toUpperCase(),
							)
						: undefined;
					const inspectionCode =
						typeof mappedInspectionId === "number"
							? (inspectionCodeById.get(mappedInspectionId) ?? "")
							: "";

					switch (key) {
						case "lp":
							return String(rowIndex + 1);
						case "kodDecyzji":
							return item.kodDecyzji ?? "";
						case "kodZalecenia":
							return recommendationCode;
						case "inspectionLp":
							return inspectionCode;
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
				};

				const decisionHeaders = decisionColumnKeys.map(
					(key) =>
						DECISION_EXPORT_COLUMNS.find((column) => column.key === key)
							?.label ?? key,
				);
				const decisionRowsForExport = relatedDecisions.map((item, index) =>
					decisionColumnKeys.map((key) =>
						normalizeExportValue(getDecisionExportValue(item, key, index)),
					),
				);

				addWorksheetWithStyles(
					workbook,
					"Inspekcje",
					inspectionHeaders,
					inspectionRowsForExport,
				);

				if (includeRecommendations && recommendationColumnKeys.length > 0) {
					addWorksheetWithStyles(
						workbook,
						"Zalecenia",
						recommendationHeaders,
						recommendationRowsForExport,
					);
				}

				if (includeSanctions && sanctionColumnKeys.length > 0) {
					addWorksheetWithStyles(
						workbook,
						"Wnioski sankcyjne",
						sanctionHeaders,
						sanctionRowsForExport,
					);
				}

				if (includeDecisions && decisionColumnKeys.length > 0) {
					addWorksheetWithStyles(
						workbook,
						"Decyzje zobowiązujące",
						decisionHeaders,
						decisionRowsForExport,
					);
				}

				const fileName = "inspekcje-zalecenia-sankcje-decyzje.xlsx";
				await saveWorkbookAsXlsx(workbook, fileName);
			} catch (error) {
				if (error instanceof DOMException && error.name === "AbortError") {
					return;
				}

				setAddInspectionError("Nie udało się wyeksportować danych do Excela.");
			} finally {
				setIsExporting(false);
			}
		},
		[
			operatorLogin,
			filteredAndSortedInspectionRows,
			inspectionNoLetterFlagsByRowId,
			isExporting,
			visibleInspectionColumnDefinitions,
		],
	);

	const handleOpenExportConfigModal = () => {
		if (isExporting || filteredAndSortedInspectionRows.length === 0) {
			return;
		}

		setIncludeRecommendationsInExport(false);
		setIncludeSanctionsInExport(false);
		setIncludeDecisionsInExport(false);
		setActiveExportColumnsTab("recommendations");
		setIsExportConfigModalOpen(true);
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
			(includeRecommendationsInExport &&
				selectedRecommendationExportColumns.length === 0) ||
			(includeSanctionsInExport &&
				selectedSanctionExportColumns.length === 0) ||
			(includeDecisionsInExport && selectedDecisionExportColumns.length === 0)
		) {
			return;
		}

		const orderedRecommendationColumns = RECOMMENDATION_EXPORT_COLUMNS.map(
			(column) => column.key,
		).filter((key) => selectedRecommendationExportColumns.includes(key));

		const orderedSanctionColumns = SANCTION_EXPORT_COLUMNS.map(
			(column) => column.key,
		).filter((key) => selectedSanctionExportColumns.includes(key));

		const orderedDecisionColumns = DECISION_EXPORT_COLUMNS.map(
			(column) => column.key,
		).filter((key) => selectedDecisionExportColumns.includes(key));

		setIsExportConfigModalOpen(false);
		void handleExportCurrentView(
			orderedRecommendationColumns,
			orderedSanctionColumns,
			orderedDecisionColumns,
			includeRecommendationsInExport,
			includeSanctionsInExport,
			includeDecisionsInExport,
		);
	};

	const handleTeamMemberToggle = (userId: number) => {
		setTeamMemberScopeError(null);
		setOutOfScopeTeamMemberUserId(null);
		setSelectedTeamMemberIds((prev) =>
			prev.includes(userId)
				? prev.filter((item) => item !== userId)
				: [...prev, userId],
		);
	};

	const handleSetIsDataAkceptacjiNotyBrak = useCallback(
		(next: SetStateAction<boolean>) => {
			setIsDataAkceptacjiNotyBrak((prev) => {
				const resolved =
					typeof next === "function"
						? (next as (previousState: boolean) => boolean)(prev)
						: next;
				if (resolved !== prev) {
					setDidToggleDataAkceptacjiNotyBrak(true);
				}
				return resolved;
			});
		},
		[],
	);

	const handleOpenAddModal = () => {
		if (!canManageInspections) {
			setRowsError("Konto zewnętrzne ma dostęp tylko do odczytu.");
			return;
		}

		const defaultLeaderUserId =
			operatorUserId &&
			availableLeaderUsers.some((user) => user.id === operatorUserId)
				? operatorUserId
				: (availableLeaderUsers[0]?.id ?? null);

		const defaultLeaderName = defaultLeaderUserId
			? getUserDisplayName(
					availableLeaderUsers.find(
						(user) => user.id === defaultLeaderUserId,
					) ?? {
						id: defaultLeaderUserId,
						login: operatorLogin,
						displayName: operatorDisplayName,
						active: true,
						visibleOnList: true,
						canBeLeader: true,
						createdByOperator: true,
						teamId: null,
						teamName: null,
					},
				)
			: operatorDisplayName.trim() || operatorLogin.trim();

		setAddInspectionForm({
			...DEFAULT_ADD_INSPECTION_FORM,
			osobaKierujaca: defaultLeaderName,
		});
		setSelectedLeaderUserId(defaultLeaderUserId);
		setAddInspectionError(null);
		setTeamMemberScopeError(null);
		setOutOfScopeTeamMemberUserId(null);
		setSelectedInspectionScopes([]);
		setSelectedTeamMemberIds([]);
		setDataAkceptacjiNotyList([]);
		setIsDataAkceptacjiNotyBrak(false);
		setDidToggleDataAkceptacjiNotyBrak(false);
		setIsTeamPickerOpen(false);
		setEditingInspectionId(null);
		setShowRequiredInspectionFieldErrors(false);
		setVersionConflictUpdatedAt(null);
		setStatusValidationViolations([]);
		setIsStatusValidationModalOpen(false);
		setSaveLockConflict(null);
		setIsAddModalOpen(true);
	};

	const handleOpenEditModal = async () => {
		if (!canManageInspections) {
			setRowsError("Konto zewnętrzne ma dostęp tylko do odczytu.");
			return;
		}

		if (!selectedInspectionId) {
			return;
		}

		if (!selectedInspectionCanEdit) {
			setAddInspectionError("Brak uprawnień do edycji tej inspekcji.");
			return;
		}

		const rowToEdit = inspectionRows.find(
			(row) => row.id === selectedInspectionId,
		);

		if (!rowToEdit) {
			return;
		}

		await loadInspectionPeopleOptionsForEdit(rowToEdit.id);

		const nextForm = mapRowToAddForm(rowToEdit);
		const noLetterFlags = inspectionNoLetterFlagsByRowId[rowToEdit.id] ?? {
			brakDataDoreczeniaPisma: false,
			brakDataPismaZastrzezenia: false,
			brakDataWyslaniaPismaZZastrzezeniami: false,
			brakDataWplywuPisma: false,
			brakDataPismaZOdpowiedzia: false,
			brakDataWyslaniaPismaZOdpowiedzia: false,
		};
		const existingLeaderUserId =
			inspectionLeaderUserIdByRowId[rowToEdit.id] ?? null;
		setAddInspectionForm({
			...nextForm,
			...noLetterFlags,
			dataDoreczeniaPisma: noLetterFlags.brakDataDoreczeniaPisma
				? ""
				: nextForm.dataDoreczeniaPisma,
			dataPismaZastrzezenia: noLetterFlags.brakDataPismaZastrzezenia
				? ""
				: nextForm.dataPismaZastrzezenia,
			dataWyslaniaPismaZZastrzezeniami:
				noLetterFlags.brakDataWyslaniaPismaZZastrzezeniami
					? ""
					: nextForm.dataWyslaniaPismaZZastrzezeniami,
			dataWplywuPisma: noLetterFlags.brakDataWplywuPisma
				? ""
				: nextForm.dataWplywuPisma,
			dataPismaZOdpowiedzia: noLetterFlags.brakDataPismaZOdpowiedzia
				? ""
				: nextForm.dataPismaZOdpowiedzia,
			dataWyslaniaPismaZOdpowiedzia:
				noLetterFlags.brakDataWyslaniaPismaZOdpowiedzia
					? ""
					: nextForm.dataWyslaniaPismaZOdpowiedzia,
		});
		setSelectedLeaderUserId(existingLeaderUserId);
		setSelectedInspectionScopes(
			normalizeInspectionScopeValues(
				parseMultiValueField(nextForm.zakresInspekcji),
			),
		);
		setSelectedTeamMemberIds(
			inspectionTeamMemberIdsByRowId[rowToEdit.id] ?? [],
		);
		const acceptanceDates =
			inspectionAcceptanceDatesByRowId[rowToEdit.id] ?? [];
		const noAcceptanceDatesFlags =
			inspectionNoAcceptanceDatesByRowId[rowToEdit.id] ?? {
				brakDatAkceptacjiNoty: false,
			};
		setDataAkceptacjiNotyList(acceptanceDates);
		setIsDataAkceptacjiNotyBrak(noAcceptanceDatesFlags.brakDatAkceptacjiNoty);
		setDidToggleDataAkceptacjiNotyBrak(false);
		setAddInspectionError(null);
		setTeamMemberScopeError(null);
		setOutOfScopeTeamMemberUserId(null);
		setIsTeamPickerOpen(false);
		setEditingInspectionId(rowToEdit.id);
		setShowRequiredInspectionFieldErrors(false);
		setVersionConflictUpdatedAt(null);
		setStatusValidationViolations([]);
		setIsStatusValidationModalOpen(false);
		setSaveLockConflict(null);
		setIsAddModalOpen(true);
	};

	const handleOpenDeleteConfirmModal = () => {
		if (!isDirector || !selectedInspectionId) {
			return;
		}

		setRowsError(null);
		setIsDeleteConfirmModalOpen(true);
	};

	const handleOpenPreviewModal = (inspectionId: string) => {
		const rowToPreview = inspectionRows.find((row) => row.id === inspectionId);
		if (!rowToPreview) {
			return;
		}

		const nextForm = mapRowToAddForm(rowToPreview);
		const noLetterFlags = inspectionNoLetterFlagsByRowId[rowToPreview.id] ?? {
			brakDataDoreczeniaPisma: false,
			brakDataPismaZastrzezenia: false,
			brakDataWyslaniaPismaZZastrzezeniami: false,
			brakDataWplywuPisma: false,
			brakDataPismaZOdpowiedzia: false,
			brakDataWyslaniaPismaZOdpowiedzia: false,
		};
		const existingLeaderUserId =
			inspectionLeaderUserIdByRowId[rowToPreview.id] ?? null;

		setAddInspectionForm({
			...nextForm,
			...noLetterFlags,
			dataDoreczeniaPisma: noLetterFlags.brakDataDoreczeniaPisma
				? ""
				: nextForm.dataDoreczeniaPisma,
			dataPismaZastrzezenia: noLetterFlags.brakDataPismaZastrzezenia
				? ""
				: nextForm.dataPismaZastrzezenia,
			dataWyslaniaPismaZZastrzezeniami:
				noLetterFlags.brakDataWyslaniaPismaZZastrzezeniami
					? ""
					: nextForm.dataWyslaniaPismaZZastrzezeniami,
			dataWplywuPisma: noLetterFlags.brakDataWplywuPisma
				? ""
				: nextForm.dataWplywuPisma,
			dataPismaZOdpowiedzia: noLetterFlags.brakDataPismaZOdpowiedzia
				? ""
				: nextForm.dataPismaZOdpowiedzia,
			dataWyslaniaPismaZOdpowiedzia:
				noLetterFlags.brakDataWyslaniaPismaZOdpowiedzia
					? ""
					: nextForm.dataWyslaniaPismaZOdpowiedzia,
		});
		setSelectedLeaderUserId(existingLeaderUserId);
		setSelectedInspectionScopes(
			normalizeInspectionScopeValues(
				parseMultiValueField(nextForm.zakresInspekcji),
			),
		);
		setSelectedTeamMemberIds(inspectionTeamMemberIdsByRowId[rowToPreview.id] ?? []);
		const acceptanceDates = inspectionAcceptanceDatesByRowId[rowToPreview.id] ?? [];
		const noAcceptanceDatesFlags =
			inspectionNoAcceptanceDatesByRowId[rowToPreview.id] ?? {
				brakDatAkceptacjiNoty: false,
			};
		setDataAkceptacjiNotyList(acceptanceDates);
		setIsDataAkceptacjiNotyBrak(noAcceptanceDatesFlags.brakDatAkceptacjiNoty);
		setDidToggleDataAkceptacjiNotyBrak(false);
		setAddInspectionError(null);
		setTeamMemberScopeError(null);
		setOutOfScopeTeamMemberUserId(null);
		setIsTeamPickerOpen(false);
		setEditingInspectionId(rowToPreview.id);
		setShowRequiredInspectionFieldErrors(false);
		setVersionConflictUpdatedAt(null);
		setStatusValidationViolations([]);
		setIsStatusValidationModalOpen(false);
		setSaveLockConflict(null);
		setIsPreviewMode(true);
		setIsAddModalOpen(true);
	};

	const handleStartEditFromPreview = useCallback(() => {
		if (!editingInspectionId) {
			return;
		}

		if (!canManageInspections) {
			setAddInspectionError("Konto zewnętrzne ma dostęp tylko do odczytu.");
			return;
		}

		if (!(inspectionCanEditByRowId[editingInspectionId] ?? false)) {
			setAddInspectionError("Brak uprawnień do edycji tej inspekcji.");
			return;
		}

		void loadInspectionPeopleOptionsForEdit(editingInspectionId);
		setAddInspectionError(null);
		setSaveLockConflict(null);
		setVersionConflictUpdatedAt(null);
		setStatusValidationViolations([]);
		setIsStatusValidationModalOpen(false);
		setShowRequiredInspectionFieldErrors(false);
		setIsPreviewMode(false);
	}, [
		canManageInspections,
		editingInspectionId,
		inspectionCanEditByRowId,
		loadInspectionPeopleOptionsForEdit,
	]);

	const getDeleteRelatedCount = (
		payload: Record<string, unknown>,
		keys: string[],
	): number | null => {
		for (const key of keys) {
			const value = payload[key];
			if (typeof value === "number" && Number.isFinite(value)) {
				return value;
			}

			if (typeof value === "string") {
				const parsed = Number(value.trim());
				if (Number.isFinite(parsed) && parsed >= 0) {
					return parsed;
				}
			}
		}

		return null;
	};

	const handleDeleteInspection = async () => {
		if (!isDirector || !selectedInspectionId || isDeletingInspection) {
			return;
		}

		setIsDeletingInspection(true);
		setRowsError(null);

		try {
			const deleteResponse = await fetch(
				`${INSPECTIONS_API_URL}/${selectedInspectionId}`,
				{
					method: "DELETE",
					headers: {
						"Content-Type": "application/json",
						"X-Operator-Login": operatorLogin,
					},
				},
			);

			if (!deleteResponse.ok) {
				const apiMessage = await getInspectionApiErrorMessage(
					deleteResponse,
					"Nie udało się usunąć inspekcji",
				);
				throw new Error(apiMessage);
			}

			let deletedRecommendationsCount: number | null = null;
			let deletedSanctionRequestsCount: number | null = null;
			let deletedObligatingDecisionsCount: number | null = null;

			const contentType = deleteResponse.headers.get("content-type") ?? "";
			if (contentType.includes("application/json")) {
				const payload = (await deleteResponse.json()) as Record<
					string,
					unknown
				>;
				const nestedDeleted =
					typeof payload.deleted === "object" && payload.deleted !== null
						? (payload.deleted as Record<string, unknown>)
						: null;

				deletedRecommendationsCount =
					getDeleteRelatedCount(payload, [
						"deletedRecommendations",
						"recommendationsDeleted",
						"deleted_recommendations",
						"recommendations_deleted",
						"deletedZalecenia",
						"zaleceniaDeleted",
						"deleted_zalecenia",
						"zalecenia",
					]) ??
					(nestedDeleted
						? getDeleteRelatedCount(nestedDeleted, [
								"recommendations",
								"recommendation",
								"recommendationCount",
								"recommendations_count",
								"zalecenia_count",
								"zalecenia",
							])
						: null);

				deletedSanctionRequestsCount =
					getDeleteRelatedCount(payload, [
						"deletedSanctionRequests",
						"sanctionRequestsDeleted",
						"deletedSanctions",
						"sanctionsDeleted",
						"deleted_sanction_requests",
						"sanction_requests_deleted",
						"deleted_sanctions",
						"sanctions_deleted",
						"deletedWnioskiSankcyjne",
						"wnioskiSankcyjneDeleted",
						"deleted_wnioski_sankcyjne",
						"wnioskiSankcyjne",
						"wnioski_sankcyjne",
						"sanctions",
						"sanctionsCount",
						"sanctions_count",
					]) ??
					(nestedDeleted
						? getDeleteRelatedCount(nestedDeleted, [
								"sanctionRequests",
								"sanctionRequest",
								"sanction_requests",
								"sanctions",
								"sanction",
								"sanctionRequestsCount",
								"sanction_requests_count",
								"sanctionsCount",
								"sanctions_count",
								"wnioskiSankcyjneCount",
								"wnioski_sankcyjne_count",
								"wnioskiSankcyjne",
								"wnioski_sankcyjne",
							])
						: null);

				deletedObligatingDecisionsCount =
					getDeleteRelatedCount(payload, [
						"deletedObligatingDecisions",
						"obligatingDecisionsDeleted",
						"deletedBindingDecisions",
						"bindingDecisionsDeleted",
						"deletedDecyzjeZobowiazujace",
						"decyzjeZobowiazujaceDeleted",
						"deleted_obligating_decisions",
						"obligating_decisions_deleted",
						"deleted_binding_decisions",
						"binding_decisions_deleted",
						"deleted_decyzje_zobowiazujace",
						"decyzje_zobowiazujace_deleted",
						"decyzjeZobowiazujace",
						"decyzje_zobowiazujace",
						"obligatingDecisions",
						"obligating_decisions",
						"bindingDecisions",
						"binding_decisions",
					]) ??
					(nestedDeleted
						? getDeleteRelatedCount(nestedDeleted, [
								"obligatingDecisions",
								"obligatingDecision",
								"obligating_decisions",
								"bindingDecisions",
								"bindingDecision",
								"binding_decisions",
								"obligatingDecisionsCount",
								"obligating_decisions_count",
								"bindingDecisionsCount",
								"binding_decisions_count",
								"decyzjeZobowiazujaceCount",
								"decyzje_zobowiazujace_count",
								"decyzjeZobowiazujace",
								"decyzje_zobowiazujace",
							])
						: null);
			}

			setIsDeleteConfirmModalOpen(false);
			setSelectedInspectionId(null);
			await loadInspections();
			window.dispatchEvent(new CustomEvent(INSPECTIONS_CHANGED_EVENT));

			const recommendationsLabel =
				deletedRecommendationsCount === null
					? "0"
					: String(deletedRecommendationsCount);
			const sanctionsLabel =
				deletedSanctionRequestsCount === null
					? "0"
					: String(deletedSanctionRequestsCount);
			const decisionsLabel =
				deletedObligatingDecisionsCount === null
					? "0"
					: String(deletedObligatingDecisionsCount);

			setDeleteSuccessModalMessage(
				`Usunięto inspekcję (zalecenia: ${recommendationsLabel}, wnioski sankcyjne: ${sanctionsLabel}, decyzje zobowiązujące: ${decisionsLabel})`,
			);
			setIsDeleteSuccessModalOpen(true);
		} catch (error) {
			setRowsError(
				error instanceof Error && error.message
					? error.message
					: "Nie udało się usunąć inspekcji.",
			);
		} finally {
			setIsDeletingInspection(false);
		}
	};

	const closeInspectionFormModal = () => {
		const shouldReloadPeopleOptions = Boolean(editingInspectionId);

		if (editInspectionLock.lockToken) {
			void editInspectionLock.release();
		}

		setIsAddModalOpen(false);
		setIsPreviewMode(false);
		setIsTeamPickerOpen(false);
		setEditingInspectionId(null);
		setSelectedInspectionScopes([]);
		setSelectedTeamMemberIds([]);
		setSelectedLeaderUserId(null);
		setDataAkceptacjiNotyList([]);
		setIsDataAkceptacjiNotyBrak(false);
		setDidToggleDataAkceptacjiNotyBrak(false);
		setVersionConflictUpdatedAt(null);
		setStatusValidationViolations([]);
		setIsStatusValidationModalOpen(false);
		setSaveLockConflict(null);
		setShowRequiredInspectionFieldErrors(false);
		setTeamMemberScopeError(null);
		setOutOfScopeTeamMemberUserId(null);

		if (shouldReloadPeopleOptions) {
			void loadInspectionDictionaries();
		}
	};
	closeInspectionFormModalRef.current = closeInspectionFormModal;

	const handleRefreshAfterConflict = async () => {
		await loadInspections();
		closeInspectionFormModal();
		setVersionConflictUpdatedAt(null);
		setStatusValidationViolations([]);
		setIsStatusValidationModalOpen(false);
		setSaveLockConflict(null);
		setAddInspectionError(null);
	};

	const handleAddInspection = async (
		event: React.FormEvent<HTMLFormElement>,
	) => {
		event.preventDefault();

		if (!canManageInspections) {
			setAddInspectionError("Konto zewnętrzne ma dostęp tylko do odczytu.");
			return;
		}

		if (shouldShowLockedByOtherUser) {
			setAddInspectionError(
				"Nie możesz teraz edytować tego wpisu, ponieważ jest edytowany przez innego użytkownika.",
			);
			return;
		}

		const isRequiredInspectionTypeMissing = !addInspectionForm.typInspekcji.trim();
		const isRequiredEntityNameMissing = !addInspectionForm.nazwaPodmiotu.trim();
		const isRequiredStartDateMissing = !addInspectionForm.poczatekInspekcji;
		const isRequiredEndDateMissing = !addInspectionForm.koniecInspekcji;
		const isRequiredStatusMissing = !addInspectionForm.status.trim();
		const hasMissingRequiredFields =
			isRequiredInspectionTypeMissing ||
			isRequiredEntityNameMissing ||
			isRequiredStartDateMissing ||
			isRequiredEndDateMissing ||
			isRequiredStatusMissing;

		setShowRequiredInspectionFieldErrors(true);

		if (hasMissingRequiredFields) {
			setAddInspectionError(null);
			return;
		}

		setShowRequiredInspectionFieldErrors(false);

		if (
			addInspectionForm.koniecInspekcji < addInspectionForm.poczatekInspekcji
		) {
			setAddInspectionError(
				"Data końca inspekcji nie może być wcześniejsza niż data początku.",
			);
			return;
		}

		const lockedEditLeaderUserId =
			editingInspectionId && !canChangeLeaderSelection
				? (inspectionLeaderUserIdByRowId[editingInspectionId] ?? selectedLeaderUserId)
				: null;
		const leaderUserIdForSave =
			lockedEditLeaderUserId ??
			(authRole === "inspector" && !editingInspectionId && operatorUserId
				? operatorUserId
				: selectedLeaderUserId);

		if (!leaderUserIdForSave) {
			setAddInspectionError("Wybierz osobę kierującą.");
			return;
		}

		const allowedLeaderIds = new Set(availableLeaderUsers.map((user) => user.id));
		const isKeepingCurrentLeaderInEdit =
			Boolean(editingInspectionId) &&
			currentEditingInspectionLeaderUserId !== null &&
			leaderUserIdForSave === currentEditingInspectionLeaderUserId;
		if (
			!(editingInspectionId && !canChangeLeaderSelection) &&
			!isKeepingCurrentLeaderInEdit &&
			!allowedLeaderIds.has(leaderUserIdForSave)
		) {
			setAddInspectionError(
				"Wybrana osoba kierująca jest poza listą dozwolonych użytkowników.",
			);
			return;
		}

		const toNullable = (value: string) => {
			const normalized = value.trim();
			return normalized ? normalized : null;
		};

		const normalizeDateList = (list: string[]) => {
			const normalized = list
				.map((value) => toDateInputValue(value))
				.filter(Boolean);
			return Array.from(new Set(normalized)).sort((left, right) =>
				left.localeCompare(right),
			);
		};

		const buildInspectionWritePayload = (
			formState: AddInspectionForm,
			teamMemberIds: number[],
			acceptanceDates: string[],
			isNoAcceptanceDates: boolean,
		) => {
			const normalizedDataAkceptacjiNotyList = isNoAcceptanceDates
				? []
				: normalizeDateList(acceptanceDates);
			const brakDatAkceptacjiNoty =
				isNoAcceptanceDates && normalizedDataAkceptacjiNotyList.length === 0;
			const dataDoreczeniaPisma = toNullable(formState.dataDoreczeniaPisma);
			const dataPismaZastrzezenia = toNullable(formState.dataPismaZastrzezenia);
			const dataWyslaniaPismaZZastrzezeniami = toNullable(
				formState.dataWyslaniaPismaZZastrzezeniami,
			);
			const dataWplywuPisma = toNullable(formState.dataWplywuPisma);
			const dataPismaZOdpowiedzia = toNullable(formState.dataPismaZOdpowiedzia);
			const dataWyslaniaPismaZOdpowiedzia = toNullable(
				formState.dataWyslaniaPismaZOdpowiedzia,
			);
			const brakDataDoreczeniaPisma =
				formState.brakDataDoreczeniaPisma && !dataDoreczeniaPisma;
			const brakDataPismaZastrzezenia =
				formState.brakDataPismaZastrzezenia && !dataPismaZastrzezenia;
			const brakDataWyslaniaPismaZZastrzezeniami =
				formState.brakDataWyslaniaPismaZZastrzezeniami &&
				!dataWyslaniaPismaZZastrzezeniami;
			const brakDataWplywuPisma =
				formState.brakDataWplywuPisma && !dataWplywuPisma;
			const brakDataPismaZOdpowiedzia =
				formState.brakDataPismaZOdpowiedzia && !dataPismaZOdpowiedzia;
			const brakDataWyslaniaPismaZOdpowiedzia =
				formState.brakDataWyslaniaPismaZOdpowiedzia &&
				!dataWyslaniaPismaZOdpowiedzia;

			return {
				nazwaPodmiotu: formState.nazwaPodmiotu.trim(),
				typInspekcji: toNullable(formState.typInspekcji),
				zakresInspekcji: toNullable(formState.zakresInspekcji),
				szczegolyDotyczaceZakresu: toNullable(
					formState.szczegolyDotyczaceZakresu,
				),
				aspektKonsumencki: toNullable(formState.aspektKonsumencki),
				poczatekInspekcji: formState.poczatekInspekcji,
				koniecInspekcji: formState.koniecInspekcji,
				rynek: toNullable(formState.rynek),
				rodzajPodmiotu: toNullable(formState.rodzajPodmiotu),
				dataProtokolu: toNullable(formState.dataProtokolu),
				dataDoreczeniaProtokolu: toNullable(formState.dataDoreczeniaProtokolu),
				dataAkceptacjiSprawozdania: toNullable(
					formState.dataAkceptacjiSprawozdania,
				),
				dataDoreczeniaPisma: brakDataDoreczeniaPisma ? null : dataDoreczeniaPisma,
				brakDataDoreczeniaPisma,
				dataPismaZastrzezenia: brakDataPismaZastrzezenia
					? null
					: dataPismaZastrzezenia,
				brakDataPismaZastrzezenia,
				dataWyslaniaPismaZZastrzezeniami:
					brakDataWyslaniaPismaZZastrzezeniami
						? null
						: dataWyslaniaPismaZZastrzezeniami,
				brakDataWyslaniaPismaZZastrzezeniami,
				dataWplywuPisma: brakDataWplywuPisma ? null : dataWplywuPisma,
				brakDataWplywuPisma,
				dataPismaZOdpowiedzia: brakDataPismaZOdpowiedzia
					? null
					: dataPismaZOdpowiedzia,
				brakDataPismaZOdpowiedzia,
				dataWyslaniaPismaZOdpowiedzia: brakDataWyslaniaPismaZOdpowiedzia
					? null
					: dataWyslaniaPismaZOdpowiedzia,
				brakDataWyslaniaPismaZOdpowiedzia,
				dataAkceptacjiNotyList: normalizedDataAkceptacjiNotyList,
				brakDatAkceptacjiNoty,
				status: toNullable(formState.status),
				komentarz: toNullable(formState.komentarz),
				teamMemberUserIds: [...teamMemberIds].sort((left, right) => left - right),
			};
		};

		const toComparablePayload = (
			payload: ReturnType<typeof buildInspectionWritePayload>,
			leaderUserId: number | null,
		) =>
			JSON.stringify({
				...payload,
				dataAkceptacjiNotyList: [...payload.dataAkceptacjiNotyList].sort(
					(left, right) => left.localeCompare(right),
				),
				osobaKierujacaUserId: leaderUserId,
			});

		const inspectionWritePayload = buildInspectionWritePayload(
			addInspectionForm,
			selectedTeamMemberIds,
			dataAkceptacjiNotyList,
			isDataAkceptacjiNotyBrak,
		);

		if (editingInspectionId) {
			const rowToEdit = inspectionRows.find((row) => row.id === editingInspectionId);
			if (rowToEdit) {
				const baseNoLetterFlags = inspectionNoLetterFlagsByRowId[rowToEdit.id] ?? {
					brakDataDoreczeniaPisma: false,
					brakDataPismaZastrzezenia: false,
					brakDataWyslaniaPismaZZastrzezeniami: false,
					brakDataWplywuPisma: false,
					brakDataPismaZOdpowiedzia: false,
					brakDataWyslaniaPismaZOdpowiedzia: false,
				};
				const baseForm = {
					...mapRowToAddForm(rowToEdit),
					...baseNoLetterFlags,
					dataDoreczeniaPisma: baseNoLetterFlags.brakDataDoreczeniaPisma
						? ""
						: toDateInputValue(rowToEdit.dataDoreczeniaPisma),
					dataPismaZastrzezenia: baseNoLetterFlags.brakDataPismaZastrzezenia
						? ""
						: toDateInputValue(rowToEdit.dataPismaZastrzezenia),
					dataWyslaniaPismaZZastrzezeniami:
						baseNoLetterFlags.brakDataWyslaniaPismaZZastrzezeniami
							? ""
							: toDateInputValue(rowToEdit.dataWyslaniaPismaZZastrzezeniami),
					dataWplywuPisma: baseNoLetterFlags.brakDataWplywuPisma
						? ""
						: toDateInputValue(rowToEdit.dataWplywuPisma),
					dataPismaZOdpowiedzia: baseNoLetterFlags.brakDataPismaZOdpowiedzia
						? ""
						: toDateInputValue(rowToEdit.dataPismaZOdpowiedzia),
					dataWyslaniaPismaZOdpowiedzia:
						baseNoLetterFlags.brakDataWyslaniaPismaZOdpowiedzia
							? ""
							: toDateInputValue(rowToEdit.dataWyslaniaPismaZOdpowiedzia),
				};
				const baseTeamMemberIds =
					inspectionTeamMemberIdsByRowId[rowToEdit.id] ?? [];
				const baseAcceptanceDates =
					inspectionAcceptanceDatesByRowId[rowToEdit.id] ?? [];
				const baseNoAcceptanceDates =
					inspectionNoAcceptanceDatesByRowId[rowToEdit.id]?.brakDatAkceptacjiNoty ??
					false;
				const baseLeaderUserId =
					inspectionLeaderUserIdByRowId[rowToEdit.id] ?? operatorUserId ?? null;
				const basePayload = buildInspectionWritePayload(
					baseForm,
					baseTeamMemberIds,
					baseAcceptanceDates,
					baseNoAcceptanceDates,
				);
				const isPayloadUnchanged =
					toComparablePayload(basePayload, baseLeaderUserId) ===
					toComparablePayload(inspectionWritePayload, leaderUserIdForSave);

				if (isPayloadUnchanged) {
					setAddInspectionError("Brak zmian do zapisania.");
					return;
				}
			}
		}

		setIsSubmittingInspection(true);
		setAddInspectionError(null);
		setTeamMemberScopeError(null);
		setOutOfScopeTeamMemberUserId(null);
		setVersionConflictUpdatedAt(null);
		setStatusValidationViolations([]);
		setIsStatusValidationModalOpen(false);
		setSaveLockConflict(null);

		try {
			if (editingInspectionId) {
				const expectedUpdatedAt =
					inspectionUpdatedAtByRowId[editingInspectionId] ?? null;
				const updateResponse = await fetch(
					`${INSPECTIONS_API_URL}/${editingInspectionId}`,
					{
						method: "PUT",
						headers: {
							"Content-Type": "application/json",
							"X-Operator-Login": operatorLogin,
						},
						body: JSON.stringify({
							...inspectionWritePayload,
							osobaKierujacaUserId: leaderUserIdForSave,
							expectedUpdatedAt,
							lockToken: editInspectionLock.lockToken,
						}),
					},
				);

				if (!updateResponse.ok) {
					if (updateResponse.status === 423) {
						let lockCode = "";
						let lockReason = "";
						let lockConflict: InspectionLockConflict | null = null;
						try {
							const payload = (await updateResponse.json()) as Record<
								string,
								unknown
							>;
							lockCode = typeof payload.code === "string" ? payload.code : "";
							lockReason =
								typeof payload.reason === "string" ? payload.reason : "";
							lockConflict = {
								ownerLogin:
									typeof payload.ownerLogin === "string"
										? payload.ownerLogin
										: "",
								ownerDisplayName:
									typeof payload.ownerDisplayName === "string"
										? payload.ownerDisplayName
										: "",
								acquiredAt:
									typeof payload.acquiredAt === "string"
										? payload.acquiredAt
										: "",
							};
						} catch {
							lockCode = "";
							lockReason = "";
							lockConflict = null;
						}

						if (lockCode === "RECORD_LOCKED") {
							setSaveLockConflict(lockConflict);
							setAddInspectionError(
								"Nie możesz teraz edytować tego wpisu, ponieważ jest edytowany przez innego użytkownika.",
							);
							return;
						}

						setSaveLockConflict(null);
						if (
							lockCode === "LOCK_REQUIRED" ||
							lockReason === "lock_required"
						) {
							setAddInspectionError(
								"Do zapisu wymagana jest aktywna blokada rekordu. Odśwież dane i otwórz formularz ponownie.",
							);
							return;
						}

						if (
							lockCode === "LOCK_TOKEN_INVALID" ||
							lockReason === "lock_token_invalid"
						) {
							setAddInspectionError(
								"Blokada edycji wygasła lub jest nieprawidłowa. Odśwież dane i otwórz formularz ponownie.",
							);
							return;
						}

						setAddInspectionError("Błąd blokady rekordu.");
						return;
					}

					if (updateResponse.status === 409) {
						const statusValidationViolations =
							await readInspectionStatusValidationViolations(updateResponse);
						if (statusValidationViolations) {
							setVersionConflictUpdatedAt(null);
							setStatusValidationViolations(statusValidationViolations);
							setIsStatusValidationModalOpen(true);
							setAddInspectionError(null);
							return;
						}

						let currentUpdatedAt: string | null = null;
						try {
							const payload = (await updateResponse.json()) as Record<
								string,
								unknown
							>;
							currentUpdatedAt =
								typeof payload.currentUpdatedAt === "string"
									? payload.currentUpdatedAt
									: typeof payload.updatedAt === "string"
										? payload.updatedAt
										: null;
						} catch {
							currentUpdatedAt = null;
						}

						setVersionConflictUpdatedAt(currentUpdatedAt);
						setAddInspectionError(
							"Dane zostały zmienione przez innego użytkownika. Odśwież widok i spróbuj ponownie.",
						);
						return;
					}

					const apiMessage = await getInspectionApiErrorMessage(
						updateResponse,
						"Nie udało się zapisać zmian",
					);
					const domainError = await readInspectionDomainError(updateResponse);
					if (
						updateResponse.status === 403 &&
						domainError?.code === "MEMBER_OUT_OF_SCOPE"
					) {
						setAddInspectionError(null);
						setTeamMemberScopeError(
							domainError.detail ||
								"Wskazana osoba w składzie zespołu jest poza zakresem operatora.",
						);
						setOutOfScopeTeamMemberUserId(domainError.memberUserId);
						return;
					}
					throw new Error(apiMessage);
				}

				await loadInspections();
				window.dispatchEvent(new CustomEvent(INSPECTIONS_CHANGED_EVENT));
				setSelectedInspectionId(editingInspectionId);
				setCreateSuccessEntityName(inspectionWritePayload.nazwaPodmiotu);
				setCreateSuccessMode("edit");
				closeInspectionFormModal();
				setIsCreateSuccessModalOpen(true);
				return;
			}

			const createResponse = await fetch(INSPECTIONS_API_URL, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					"X-Operator-Login": operatorLogin,
				},
				body: JSON.stringify({
					...inspectionWritePayload,
					osobaKierujacaUserId: leaderUserIdForSave,
				}),
			});

			if (!createResponse.ok) {
				if (createResponse.status === 409) {
					const statusValidationViolations =
						await readInspectionStatusValidationViolations(createResponse);
					if (statusValidationViolations) {
						setVersionConflictUpdatedAt(null);
						setStatusValidationViolations(statusValidationViolations);
						setIsStatusValidationModalOpen(true);
						setAddInspectionError(null);
						return;
					}
				}

				const domainError = await readInspectionDomainError(createResponse);
				if (
					createResponse.status === 403 &&
					domainError?.code === "MEMBER_OUT_OF_SCOPE"
				) {
					setAddInspectionError(null);
					setTeamMemberScopeError(
						domainError.detail ||
							"Wskazana osoba w składzie zespołu jest poza zakresem operatora.",
					);
					setOutOfScopeTeamMemberUserId(domainError.memberUserId);
					return;
				}
				const apiMessage = await getInspectionApiErrorMessage(
					createResponse,
					"Nie udało się dodać rekordu",
				);
				throw new Error(apiMessage);
			}

			let createdRecordId: string | null = null;
			const contentType = createResponse.headers.get("content-type") ?? "";
			if (contentType.includes("application/json")) {
				const createdRecord =
					(await createResponse.json()) as Partial<InspectionRow>;
				createdRecordId =
					typeof createdRecord.id === "string" ? createdRecord.id : null;
			}

			await loadInspections();
			window.dispatchEvent(new CustomEvent(INSPECTIONS_CHANGED_EVENT));
			handlePageChange(1);
			if (createdRecordId) {
				setSelectedInspectionId(createdRecordId);
			}
			setCreateSuccessEntityName(inspectionWritePayload.nazwaPodmiotu);
			setCreateSuccessMode("create");
			closeInspectionFormModal();
			setIsCreateSuccessModalOpen(true);
		} catch (error) {
			setAddInspectionError(
				error instanceof Error && error.message
					? error.message
					: "Nie udało się zapisać rekordu. Sprawdź połączenie z backendem.",
			);
		} finally {
			setIsSubmittingInspection(false);
		}
	};

	return (
		<section
			className="rounded-2xl border border-slate-700/70 bg-[#101f39] p-4 sm:p-5"
			onClick={(event) => {
				if (event.target === event.currentTarget) {
					setSelectedInspectionId(null);
				}
				event.stopPropagation();
			}}
		>
			<TablePanelToolbar
				title="Inspekcje"
				canClearFilters={canClearFilters}
				canResetColumnWidths={hasCustomColumnWidths}
				isExporting={isExporting}
				hasRowsToExport={filteredAndSortedInspectionRows.length > 0}
				onOpenViewModal={handleOpenInspectionViewModal}
				onClearFilters={clearFilters}
				onResetColumnWidths={handleResetColumnWidths}
				onExport={handleOpenExportConfigModal}
				actions={
					<>
						{canManageInspections ? (
							<>
								<button
									type="button"
									onClick={handleOpenAddModal}
									className="inline-flex h-10 items-center gap-2 rounded-lg border border-[#8ec5a1] bg-[#b9e8c9] px-3.5 font-semibold text-[#1f5130] text-sm shadow-[inset_0_1px_0_rgba(255,255,255,0.45)] transition-colors hover:bg-[#a5debb]"
								>
									<Plus size={15} />
									Dodaj inspekcję
								</button>

								<button
									type="button"
									disabled={!selectedInspectionId || !selectedInspectionCanEdit}
									onClick={() => {
										void handleOpenEditModal();
									}}
									className="inline-flex h-10 items-center gap-2 rounded-lg border px-3.5 font-semibold text-sm transition-colors enabled:border-[#93b9ee] enabled:bg-[#d9e9ff] enabled:text-[#21508f] enabled:hover:bg-[#c9e0ff] disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-[#1a2946] disabled:text-slate-500"
								>
									<Pencil size={15} />
									Edytuj
								</button>
							</>
						) : null}

						{isDirector ? (
							<button
								type="button"
								disabled={!selectedInspectionId || isDeletingInspection}
								onClick={handleOpenDeleteConfirmModal}
								className="inline-flex h-10 items-center gap-2 rounded-lg border px-3.5 font-semibold text-sm transition-colors enabled:border-[#f2a3a3] enabled:bg-[#6f2a36] enabled:text-[#ffe5e8] enabled:hover:bg-[#833242] disabled:cursor-not-allowed disabled:border-slate-300 disabled:bg-slate-200 disabled:text-slate-500"
							>
								<Trash2 size={15} />
								{isDeletingInspection ? "Usuwanie..." : "Usuń"}
							</button>
						) : null}
					</>
				}
			/>

			<InspectionsDataTable
				rowsError={rowsError}
				isRowsLoading={isRowsLoading}
				visibleColumns={visibleInspectionColumnDefinitions}
				columnWidths={columnWidths}
				minColumnWidth={INSPECTIONS_MIN_COLUMN_WIDTH}
				sortColumnKey={sortColumnKey}
				sortDirection={sortDirection}
				advancedFilters={advancedFilters}
				columnFilters={columnFilters}
				rows={paginatedInspectionRows}
				noAcceptanceDatesByRowId={inspectionNoAcceptanceDatesByRowId}
				noLetterFlagsByRowId={inspectionNoLetterFlagsByRowId}
				selectedInspectionId={selectedInspectionId}
				flashInspectionId={flashInspectionId}
				onSelectInspection={setSelectedInspectionId}
				onOpenInspectionPreview={handleOpenPreviewModal}
				onSortByColumn={handleSortByColumn}
				onResizeColumn={handleResizeColumn}
				onOpenAdvancedFilter={openAdvancedFilterForColumn}
				onFilterChange={handleFilterChange}
				footer={
					<TablePagination
						currentPage={currentPage}
						totalPages={totalPages}
						paginationItems={paginationItems}
						totalItems={filteredAndSortedInspectionRows.length}
						pageSize={pageSize}
						onPageChange={handlePageChange}
						pageSizeOptions={TABLE_PAGE_SIZE_OPTIONS}
						onPageSizeChange={handlePageSizeChange}
						showWhenSinglePage
					/>
				}
			/>

			<TableColumnPickerModal<InspectionColumnKey, InspectionViewId>
				isOpen={isColumnPickerOpen}
				layoutOptions={INSPECTION_VIEW_OPTIONS}
				selectedLayoutId={draftSelectedInspectionView}
				onSelectLayout={handleDraftViewSelect}
				columns={draftSelectableColumnDefinitions}
				hiddenColumns={draftHiddenColumns}
				visibleColumnsCount={draftVisibleInspectionColumnsCount}
				onClose={() => setIsColumnPickerOpen(false)}
				onChangeColumnVisibility={handleDraftColumnVisibilityChange}
				onChangeColumnDisplayMode={(columnKey, value) => {
					if (!isInspectionNameVariantColumnKey(columnKey)) {
						return;
					}

					if (
						!isInspectionNameVariant(value) ||
						!isInspectionNameVariantAllowedForColumn(columnKey, value)
					) {
						return;
					}

					setDraftInspectionNameVariants((prev) => ({
						...prev,
						[columnKey]: value,
					}));
				}}
				columnDisplayModeOptions={columnDisplayModeOptionsByKey}
				columnDisplayModeValues={draftColumnDisplayModeValuesByKey}
				onResetSelection={handleResetInspectionViewSelection}
				onShowAllColumns={handleDraftSelectAllColumns}
				onHideAllColumns={handleDraftDeselectAllColumns}
				onApply={handleApplyInspectionViewChanges}
			/>

			<TableAdvancedFilterModal
				isOpen={isAdvancedFilterModalOpen}
				anchor={advancedFilterAnchor}
				columnLabel={
					draftSelectableColumnDefinitions.find(
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

			<InspectionsFormModal
				isOpen={isAddModalOpen}
				isPreviewMode={isPreviewMode}
				canStartEditFromPreview={canManageInspections && previewInspectionCanEdit}
				onStartEditFromPreview={handleStartEditFromPreview}
				editingInspectionId={editingInspectionId}
				editingInspectionCode={
					editingInspectionId ? (selectedInspectionRow?.kodInspekcji ?? null) : null
				}
				showRequiredFieldErrors={showRequiredInspectionFieldErrors}
				isReadOnly={isPreviewMode || isReadOnlyDueToLock}
				isSaveDisabledDueToLock={
					isEditMode &&
					(editInspectionLock.isAcquireFailed ||
						editInspectionLock.isConnectionLost ||
						editInspectionLock.isExpired)
				}
				lockNotice={inspectionLockNotice}
				inactivityIsWarning={inactivityTimeout.isWarning}
				inactivitySecondsRemaining={inactivityTimeout.secondsRemaining}
				onInactivityContinue={inactivityTimeout.resetTimer}
				onRetryAcquire={
					isEditMode && editInspectionLock.isAcquireFailed
						? editInspectionLock.retryAcquire
						: undefined
				}
				versionConflictUpdatedAt={versionConflictUpdatedAt}
				onRefreshAfterConflict={() => {
					void handleRefreshAfterConflict();
				}}
				addInspectionForm={addInspectionForm}
				setAddInspectionForm={setAddInspectionForm}
				entityNameOptions={entityNameOptions}
				inspectionTypeOptions={inspectionTypeOptions}
				inspectionScopeOptions={inspectionScopeOptions}
				marketOptions={marketOptions}
				entityTypeOptions={entityTypeOptions}
				inspectionStatusOptions={inspectionStatusOptions}
				selectedInspectionScopes={selectedInspectionScopes}
				setSelectedInspectionScopes={setSelectedInspectionScopes}
				operatorDisplayName={operatorDisplayName}
				operatorLogin={operatorLogin}
				isTeamPickerOpen={isTeamPickerOpen}
				setIsTeamPickerOpen={setIsTeamPickerOpen}
				selectedTeamMemberIds={selectedTeamMemberIds}
				setSelectedTeamMemberIds={setSelectedTeamMemberIds}
				selectedTeamMembers={selectedTeamMembers}
				teamMemberScopeError={teamMemberScopeError}
				outOfScopeTeamMemberUserId={outOfScopeTeamMemberUserId}
				activeUsers={activeUsers}
				availableLeaderUsers={leaderOptionsForModal}
				leaderChangeIrreversibleWarning={leaderChangeIrreversibleWarning}
				forceLeaderSelectionReadonly={!canChangeLeaderSelection}
				selectedLeaderUserId={selectedLeaderUserId}
				setSelectedLeaderUserId={setSelectedLeaderUserId}
				dataAkceptacjiNotyList={dataAkceptacjiNotyList}
				setDataAkceptacjiNotyList={setDataAkceptacjiNotyList}
				isDataAkceptacjiNotyBrak={isDataAkceptacjiNotyBrak}
				setIsDataAkceptacjiNotyBrak={handleSetIsDataAkceptacjiNotyBrak}
				addInspectionError={addInspectionError}
				isSubmittingInspection={isSubmittingInspection}
				onToggleTeamMember={handleTeamMemberToggle}
				onClose={closeInspectionFormModal}
				onSubmit={handleAddInspection}
			/>

			<EntitySuccessModal
				isOpen={isCreateSuccessModalOpen}
				heading={
					createSuccessMode === "edit"
						? "Inspekcja została zaktualizowana"
						: "Inspekcja została dodana"
				}
				detailsMessage={
					createSuccessEntityName.trim()
						? `Dla podmiotu ${createSuccessEntityName.trim()}.`
						: createSuccessMode === "edit"
							? "Rekord zaktualizowano w tabeli."
							: "Rekord został dodany do tabeli."
				}
				onClose={() => {
					setIsCreateSuccessModalOpen(false);
					setCreateSuccessEntityName("");
					setCreateSuccessMode("create");
				}}
			/>

			<EntitySuccessModal
				isOpen={isDeleteSuccessModalOpen}
				heading="Inspekcja została usunięta"
				detailsMessage={
					deleteSuccessModalMessage ??
					"Inspekcja oraz powiązane rekordy zostały usunięte."
				}
				onClose={() => {
					setIsDeleteSuccessModalOpen(false);
					setDeleteSuccessModalMessage(null);
				}}
			/>

			{isDeleteConfirmModalOpen ? (
				<div className="fixed inset-0 z-60 flex items-center justify-center p-4">
					<button
						type="button"
						aria-label="Zamknij okno usuwania inspekcji"
						className="absolute inset-0 bg-slate-950/65"
						onClick={() => {
							if (isDeletingInspection) {
								return;
							}

							setIsDeleteConfirmModalOpen(false);
						}}
					/>

					<div
						role="dialog"
						aria-modal="true"
						aria-label="Potwierdzenie usunięcia inspekcji"
						className="relative z-10 w-full max-w-xl rounded-2xl border border-slate-300 bg-white p-5 text-slate-900 shadow-[0_24px_56px_rgba(2,8,23,0.35)]"
					>
						<h3 className="font-semibold text-base text-slate-900">
							Usuń inspekcję
						</h3>
						<p className="mt-2 text-slate-700 text-sm leading-6">
							Usunięcie inspekcji spowoduje trwałe usunięcie wszystkich
							powiązanych zaleceń, wniosków sankcyjnych oraz decyzji
							zobowiązujących. Czy na pewno chcesz kontynuować?
						</p>
						{selectedInspectionRow?.nazwaPodmiotu ? (
							<p className="mt-2 text-slate-500 text-xs">
								Podmiot: {selectedInspectionRow.nazwaPodmiotu}
							</p>
						) : null}

						<div className="mt-5 flex items-center justify-end gap-2">
							<button
								type="button"
								onClick={() => setIsDeleteConfirmModalOpen(false)}
								disabled={isDeletingInspection}
								className="inline-flex h-10 items-center rounded-lg border border-slate-300 bg-white px-4 font-semibold text-slate-700 text-sm transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
							>
								Anuluj
							</button>
							<button
								type="button"
								onClick={() => void handleDeleteInspection()}
								disabled={isDeletingInspection}
								className="inline-flex h-10 items-center rounded-lg border border-[#f2a3a3] bg-[#6f2a36] px-4 font-semibold text-[#ffe5e8] text-sm transition-colors hover:bg-[#833242] disabled:cursor-not-allowed disabled:opacity-60"
							>
								{isDeletingInspection ? "Usuwanie..." : "Usuń inspekcję"}
							</button>
						</div>
					</div>
				</div>
			) : null}

			<ExportConfigModal
				isOpen={isExportConfigModalOpen}
				description="Inspekcje eksportują aktualny widok tabeli. Wybierz kolumny dla zakładek powiązanych."
				relationsLabel="Powiąż wybrane inspekcje z:"
				relations={[
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
						id: "sanctions",
						label: "Wnioski sankcyjne",
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
						id: "sanctions",
						label: "Wnioski sankcyjne",
						columns: SANCTION_EXPORT_COLUMNS.map((column) => ({
							key: column.key,
							label: column.label,
						})),
						selectedKeys: selectedSanctionExportColumns,
						onToggleKey: (key, isSelected) =>
							toggleSanctionExportColumn(
								key as SanctionExportColumnKey,
								isSelected,
							),
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
						tabId as "recommendations" | "sanctions" | "decisions",
					)
				}
				onClose={() => setIsExportConfigModalOpen(false)}
				onConfirm={handleConfirmExportFromModal}
				isConfirmDisabled={
					isExporting ||
					(includeRecommendationsInExport &&
						selectedRecommendationExportColumns.length === 0) ||
					(includeSanctionsInExport &&
						selectedSanctionExportColumns.length === 0) ||
					(includeDecisionsInExport &&
						selectedDecisionExportColumns.length === 0)
				}
				isExporting={isExporting}
			/>

			{isStatusValidationModalOpen ? (
				<div className="fixed inset-0 z-60 flex items-center justify-center p-4">
					<button
						type="button"
						aria-label="Zamknij okno walidacji statusu"
						className="absolute inset-0 bg-slate-950/65"
						onClick={() => setIsStatusValidationModalOpen(false)}
					/>

					<div
						role="dialog"
						aria-modal="true"
						aria-label="Walidacja statusu inspekcji"
						className="relative z-10 w-full max-w-2xl rounded-2xl border border-slate-300 bg-white p-5 text-slate-900 shadow-[0_24px_56px_rgba(2,8,23,0.35)]"
					>
						<h3 className="font-semibold text-base text-slate-900">
							Nie można zapisać inspekcji z tym statusem
						</h3>
						{selectedStatusForValidation ? (
							<p className="mt-2 text-slate-800 text-sm">
								Status: <span className="font-semibold">{selectedStatusForValidation}</span>
							</p>
						) : null}

						<div className="mt-4 rounded-lg border border-rose-200 bg-rose-50 p-3">
							<p className="font-semibold text-rose-700 text-sm">Naruszenia:</p>
							<ul className="mt-2 list-disc space-y-1 pl-5 text-rose-800 text-sm">
								{statusValidationViolations.map((violation, index) => (
									<li
										key={`${String(violation.violationCodeId)}-${violation.message}-${index}`}
									>
										{violation.message}
									</li>
								))}
							</ul>
						</div>

						<div className="mt-5 flex items-center justify-end gap-2">
							<button
								type="button"
								onClick={() => setIsStatusValidationModalOpen(false)}
								className="inline-flex h-10 items-center rounded-lg border border-slate-300 bg-white px-4 font-semibold text-slate-700 text-sm transition-colors hover:bg-slate-100"
							>
								Rozumiem
							</button>
						</div>
					</div>
				</div>
			) : null}
		</section>
	);
}
