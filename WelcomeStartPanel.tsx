"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle } from "lucide-react";
import {
	Cell,
	Pie,
	PieChart,
	ResponsiveContainer,
	Tooltip,
} from "recharts";

import { getStoredAuthSession } from "@/features/auth/session";
import { normalizeAuthRole } from "@/features/auth/types";
import {
	fetchInspectionsDetailedReport,
	fetchInspectionsStageSummary,
	fetchRecommendationsDetailedReport,
	fetchRecommendationsStageSummary,
} from "@/features/reports/api";
import { TableSurface } from "@/shared/components/table/TableSurface";
import type {
	InspectionStageSummaryGroup,
	ReportInspectionDetailedRow,
	ReportRecommendationDetailedRow,
	ReportsRecommendationsStageSummaryResponse,
	ReportsInspectionsStageSummaryResponse,
} from "@/features/reports/types";

type WelcomeStartPanelProps = {
	operatorLogin: string;
};

const DASHBOARD_OPEN_INSPECTION_EVENT = "dashboard:open-inspection";
const DASHBOARD_OPEN_INSPECTION_CODE_KEY = "triangle.dashboard.openInspectionCode";
const DASHBOARD_OPEN_RECOMMENDATION_EVENT = "dashboard:open-recommendation";
const DASHBOARD_OPEN_RECOMMENDATION_CODE_KEY =
	"triangle.dashboard.openRecommendationCode";
const DASHBOARD_ACTIVE_TOP_SECTION_KEY = "triangle.dashboard.activeTopSection";
const DASHBOARD_SELECTED_INSPECTION_STAGE_FILTERS_KEY =
	"triangle.dashboard.selectedInspectionStageFilters";
const DASHBOARD_SELECTED_RECOMMENDATION_STATUSES_KEY =
	"triangle.dashboard.selectedRecommendationStatuses";
const STAGE_OVERVIEW_COLORS = ["#0f766e", "#14b8a6", "#38bdf8", "#6366f1"] as const;

function isInformationalClosedGroup(stageGroupCode: string, stageGroupLabel: string) {
	const normalizedCode = stageGroupCode.trim().toLowerCase();
	const normalizedLabel = stageGroupLabel.trim().toLowerCase();
	return (
		normalizedCode === "closed" ||
		normalizedLabel.includes("zamkniete") ||
		normalizedLabel.includes("zamknięte")
	);
}

function normalizePolishLabel(label: string) {
	return label
		.replace(/inspekcja\b/gi, "inspekcja")
		.replace(/inspekcji\b/gi, "inspekcji")
		.replace(/przed inspekcja\b/gi, "Przed inspekcją")
		.replace(/w trakcie inspekcji\b/gi, "W trakcie inspekcji")
		.replace(/po inspekcji\b/gi, "Po inspekcji")
		.replace(/rekomendacje\b/gi, "Rekomendacje")
		.replace(/wplynely\b/gi, "Wpłynęły")
		.replace(/wplynela\b/gi, "Wpłynęła")
		.replace(/zastrzezenia\b/gi, "zastrzeżenia")
		.replace(/odpowiedz\b/gi, "odpowiedź")
		.replace(/zamkniete\b/gi, "zamknięte")
		.replace(/piszemy zalecenia\b/gi, "Piszemy zalecenia")
		.replace(/pismo ustalenia\b/gi, "Pismo ustalenia");
}

type StageFilter = {
	stageGroupCode: string;
	stageGroupLabel: string;
	stageSubgroupCode: string;
	stageSubgroupLabel: string;
};

type StageOverviewSlice = {
	stageGroupCode: string;
	stageGroupLabel: string;
	count: number;
};

type RecommendationStatusFilter = {
	stageGroupCode: string;
	stageGroupLabel: string;
};

// Zmieniaj te wartości, aby ustawić startowe szerokości kolumn.
const DEFAULT_COLUMN_WIDTHS = {
	statusInspekcji: 240,
	kodInspekcji: 160,
	nazwaPodmiotu: 240,
	rodzajPodmiotu: 230,
	zakresInspekcji: 200,
	inspektorKierujacy: 240,
	poczatekInspekcji: 170,
	koniecInspekcji: 170,
} as const;

// Zmieniaj te wartości, aby ustawić minimalne szerokości kolumn.
const MIN_COLUMN_WIDTHS = {
	statusInspekcji: 160,
	kodInspekcji: 120,
	nazwaPodmiotu: 220,
	rodzajPodmiotu: 180,
	zakresInspekcji: 200,
	inspektorKierujacy: 180,
	poczatekInspekcji: 140,
	koniecInspekcji: 140,
} as const;

function resolveMinWidth(key: keyof typeof DEFAULT_COLUMN_WIDTHS) {
	// If default width is smaller than configured minimum, honor the default.
	return Math.min(MIN_COLUMN_WIDTHS[key], DEFAULT_COLUMN_WIDTHS[key]);
}

const TABLE_COLUMNS: Array<{
	key: keyof ReportInspectionDetailedRow;
	label: string;
	defaultWidth: number;
	minWidth: number;
}> = [
	{
		key: "statusInspekcji",
		label: "Status",
		defaultWidth: DEFAULT_COLUMN_WIDTHS.statusInspekcji,
		minWidth: resolveMinWidth("statusInspekcji"),
	},
	{
		key: "kodInspekcji",
		label: "Kod inspekcji",
		defaultWidth: DEFAULT_COLUMN_WIDTHS.kodInspekcji,
		minWidth: resolveMinWidth("kodInspekcji"),
	},
	{
		key: "nazwaPodmiotu",
		label: "Nazwa podmiotu",
		defaultWidth: DEFAULT_COLUMN_WIDTHS.nazwaPodmiotu,
		minWidth: resolveMinWidth("nazwaPodmiotu"),
	},
	{
		key: "rodzajPodmiotu",
		label: "Rodzaj podmiotu",
		defaultWidth: DEFAULT_COLUMN_WIDTHS.rodzajPodmiotu,
		minWidth: resolveMinWidth("rodzajPodmiotu"),
	},
	{
		key: "zakresInspekcji",
		label: "Zakres inspekcji",
		defaultWidth: DEFAULT_COLUMN_WIDTHS.zakresInspekcji,
		minWidth: resolveMinWidth("zakresInspekcji"),
	},
	{
		key: "inspektorKierujacy",
		label: "Inspektor kierujący",
		defaultWidth: DEFAULT_COLUMN_WIDTHS.inspektorKierujacy,
		minWidth: resolveMinWidth("inspektorKierujacy"),
	},
	{
		key: "poczatekInspekcji",
		label: "Początek inspekcji",
		defaultWidth: DEFAULT_COLUMN_WIDTHS.poczatekInspekcji,
		minWidth: resolveMinWidth("poczatekInspekcji"),
	},
	{
		key: "koniecInspekcji",
		label: "Koniec inspekcji",
		defaultWidth: DEFAULT_COLUMN_WIDTHS.koniecInspekcji,
		minWidth: resolveMinWidth("koniecInspekcji"),
	},
];

const INITIAL_COLUMN_WIDTHS: Record<keyof ReportInspectionDetailedRow, number> =
	TABLE_COLUMNS.reduce(
		(accumulator, column) => ({
			...accumulator,
			[column.key]: column.defaultWidth,
		}),
		{} as Record<keyof ReportInspectionDetailedRow, number>,
	);

const RECOMMENDATION_DEFAULT_COLUMN_WIDTHS = {
	status: 200,
	recommendationId: 140,
	inspectionId: 140,
	nazwaPodmiotu: 240,
	dataZalecen: 150,
	terminZalecen: 150,
	terminWykonaniaZalecen: 230,
	liczbaZalecen: 140,
} as const;

const RECOMMENDATION_MIN_COLUMN_WIDTHS = {
	status: 150,
	recommendationId: 120,
	inspectionId: 120,
	nazwaPodmiotu: 180,
	dataZalecen: 130,
	terminZalecen: 130,
	terminWykonaniaZalecen: 180,
	liczbaZalecen: 120,
} as const;

function resolveRecommendationMinWidth(
	key: keyof typeof RECOMMENDATION_DEFAULT_COLUMN_WIDTHS,
) {
	return Math.min(
		RECOMMENDATION_MIN_COLUMN_WIDTHS[key],
		RECOMMENDATION_DEFAULT_COLUMN_WIDTHS[key],
	);
}

const RECOMMENDATION_TABLE_COLUMNS: Array<{
	key: keyof ReportRecommendationDetailedRow;
	label: string;
	defaultWidth: number;
	minWidth: number;
}> = [
	{
		key: "status",
		label: "Status",
		defaultWidth: RECOMMENDATION_DEFAULT_COLUMN_WIDTHS.status,
		minWidth: resolveRecommendationMinWidth("status"),
	},
	{
		key: "recommendationId",
		label: "Id zalecenia",
		defaultWidth: RECOMMENDATION_DEFAULT_COLUMN_WIDTHS.recommendationId,
		minWidth: resolveRecommendationMinWidth("recommendationId"),
	},
	{
		key: "inspectionId",
		label: "Id inspekcji",
		defaultWidth: RECOMMENDATION_DEFAULT_COLUMN_WIDTHS.inspectionId,
		minWidth: resolveRecommendationMinWidth("inspectionId"),
	},
	{
		key: "nazwaPodmiotu",
		label: "Nazwa podmiotu",
		defaultWidth: RECOMMENDATION_DEFAULT_COLUMN_WIDTHS.nazwaPodmiotu,
		minWidth: resolveRecommendationMinWidth("nazwaPodmiotu"),
	},
	{
		key: "dataZalecen",
		label: "Data zaleceń",
		defaultWidth: RECOMMENDATION_DEFAULT_COLUMN_WIDTHS.dataZalecen,
		minWidth: resolveRecommendationMinWidth("dataZalecen"),
	},
	{
		key: "terminZalecen",
		label: "Termin zaleceń",
		defaultWidth: RECOMMENDATION_DEFAULT_COLUMN_WIDTHS.terminZalecen,
		minWidth: resolveRecommendationMinWidth("terminZalecen"),
	},
	{
		key: "terminWykonaniaZalecen",
		label: "Termin wykonania zaleceń",
		defaultWidth: RECOMMENDATION_DEFAULT_COLUMN_WIDTHS.terminWykonaniaZalecen,
		minWidth: resolveRecommendationMinWidth("terminWykonaniaZalecen"),
	},
	{
		key: "liczbaZalecen",
		label: "Liczba zaleceń",
		defaultWidth: RECOMMENDATION_DEFAULT_COLUMN_WIDTHS.liczbaZalecen,
		minWidth: resolveRecommendationMinWidth("liczbaZalecen"),
	},
];

const INITIAL_RECOMMENDATION_COLUMN_WIDTHS: Record<
	keyof ReportRecommendationDetailedRow,
	number
> = RECOMMENDATION_TABLE_COLUMNS.reduce(
	(accumulator, column) => ({
		...accumulator,
		[column.key]: column.defaultWidth,
	}),
	{} as Record<keyof ReportRecommendationDetailedRow, number>,
);

export function WelcomeStartPanel({ operatorLogin }: WelcomeStartPanelProps) {
	const [activeTopSection, setActiveTopSection] = useState<"inspections" | "recommendations">(
		"inspections",
	);
	const [rows, setRows] = useState<ReportInspectionDetailedRow[]>([]);
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [stageSummary, setStageSummary] =
		useState<ReportsInspectionsStageSummaryResponse | null>(null);
	const [isStageSummaryLoading, setIsStageSummaryLoading] = useState(true);
	const [stageSummaryError, setStageSummaryError] = useState<string | null>(null);
	const [selectedStageFilters, setSelectedStageFilters] = useState<StageFilter[]>([]);
	const [columnWidths, setColumnWidths] = useState(INITIAL_COLUMN_WIDTHS);
	const [recommendationRows, setRecommendationRows] = useState<ReportRecommendationDetailedRow[]>(
		[],
	);
	const [isRecommendationsLoading, setIsRecommendationsLoading] = useState(true);
	const [recommendationsError, setRecommendationsError] = useState<string | null>(null);
	const [recommendationSummary, setRecommendationSummary] =
		useState<ReportsRecommendationsStageSummaryResponse | null>(null);
	const [isRecommendationSummaryLoading, setIsRecommendationSummaryLoading] = useState(true);
	const [recommendationSummaryError, setRecommendationSummaryError] = useState<string | null>(
		null,
	);
	const [selectedRecommendationStatuses, setSelectedRecommendationStatuses] = useState<
		RecommendationStatusFilter[]
	>([]);
	const [recommendationColumnWidths, setRecommendationColumnWidths] = useState(
		INITIAL_RECOMMENDATION_COLUMN_WIDTHS,
	);
	const [isDashboardStateHydrated, setIsDashboardStateHydrated] = useState(false);

	const authRole = useMemo(() => {
		const storedRole = getStoredAuthSession()?.user?.rola;
		return normalizeAuthRole(storedRole);
	}, []);

	useEffect(() => {
		if (typeof window === "undefined") {
			setIsDashboardStateHydrated(true);
			return;
		}

		const savedTopSection = window.sessionStorage.getItem(
			DASHBOARD_ACTIVE_TOP_SECTION_KEY,
		);
		if (savedTopSection === "inspections" || savedTopSection === "recommendations") {
			setActiveTopSection(savedTopSection);
		}

		const savedInspectionFiltersRaw = window.sessionStorage.getItem(
			DASHBOARD_SELECTED_INSPECTION_STAGE_FILTERS_KEY,
		);
		if (savedInspectionFiltersRaw) {
			try {
				const parsed = JSON.parse(savedInspectionFiltersRaw);
				if (Array.isArray(parsed)) {
					const normalized = parsed
						.filter(
							(item) =>
								item &&
								typeof item === "object" &&
								typeof (item as StageFilter).stageGroupCode === "string" &&
								typeof (item as StageFilter).stageGroupLabel === "string" &&
								typeof (item as StageFilter).stageSubgroupCode === "string" &&
								typeof (item as StageFilter).stageSubgroupLabel === "string",
						)
						.map((item) => ({
							stageGroupCode: (item as StageFilter).stageGroupCode,
							stageGroupLabel: (item as StageFilter).stageGroupLabel,
							stageSubgroupCode: (item as StageFilter).stageSubgroupCode,
							stageSubgroupLabel: (item as StageFilter).stageSubgroupLabel,
						}));
					setSelectedStageFilters(normalized);
				}
			} catch {
				window.sessionStorage.removeItem(
					DASHBOARD_SELECTED_INSPECTION_STAGE_FILTERS_KEY,
				);
			}
		}

		const savedRecommendationFiltersRaw = window.sessionStorage.getItem(
			DASHBOARD_SELECTED_RECOMMENDATION_STATUSES_KEY,
		);
		if (savedRecommendationFiltersRaw) {
			try {
				const parsed = JSON.parse(savedRecommendationFiltersRaw);
				if (Array.isArray(parsed)) {
					const normalized = parsed
						.filter(
							(item) =>
								item &&
								typeof item === "object" &&
								typeof (item as RecommendationStatusFilter).stageGroupCode === "string" &&
								typeof (item as RecommendationStatusFilter).stageGroupLabel === "string",
						)
						.map((item) => ({
							stageGroupCode: (item as RecommendationStatusFilter).stageGroupCode,
							stageGroupLabel: (item as RecommendationStatusFilter).stageGroupLabel,
						}));
					setSelectedRecommendationStatuses(normalized);
				}
			} catch {
				window.sessionStorage.removeItem(
					DASHBOARD_SELECTED_RECOMMENDATION_STATUSES_KEY,
				);
			}
		}

		setIsDashboardStateHydrated(true);
	}, []);

	useEffect(() => {
		if (typeof window === "undefined" || !isDashboardStateHydrated) {
			return;
		}

		window.sessionStorage.setItem(DASHBOARD_ACTIVE_TOP_SECTION_KEY, activeTopSection);
	}, [activeTopSection, isDashboardStateHydrated]);

	useEffect(() => {
		if (typeof window === "undefined" || !isDashboardStateHydrated) {
			return;
		}

		window.sessionStorage.setItem(
			DASHBOARD_SELECTED_INSPECTION_STAGE_FILTERS_KEY,
			JSON.stringify(selectedStageFilters),
		);
	}, [isDashboardStateHydrated, selectedStageFilters]);

	useEffect(() => {
		if (typeof window === "undefined" || !isDashboardStateHydrated) {
			return;
		}

		window.sessionStorage.setItem(
			DASHBOARD_SELECTED_RECOMMENDATION_STATUSES_KEY,
			JSON.stringify(selectedRecommendationStatuses),
		);
	}, [isDashboardStateHydrated, selectedRecommendationStatuses]);

	const loadInspections = useCallback(async () => {
		setIsLoading(true);
		setIsStageSummaryLoading(true);
		setError(null);
		setStageSummaryError(null);

		const [detailedResult, summaryResult] = await Promise.all([
			fetchInspectionsDetailedReport(operatorLogin),
			fetchInspectionsStageSummary(operatorLogin),
		]);

		if (!detailedResult.ok) {
			setRows([]);
			setError(detailedResult.error);
		} else {
			setRows(detailedResult.data.rows);
		}

		if (!summaryResult.ok) {
			setStageSummary(null);
			setStageSummaryError(summaryResult.error);
		} else {
			setStageSummary(summaryResult.data);
		}

		setIsLoading(false);
		setIsStageSummaryLoading(false);
	}, [operatorLogin]);

	const loadRecommendations = useCallback(async () => {
		setIsRecommendationsLoading(true);
		setIsRecommendationSummaryLoading(true);
		setRecommendationsError(null);
		setRecommendationSummaryError(null);

		const [detailedResult, summaryResult] = await Promise.all([
			fetchRecommendationsDetailedReport(operatorLogin),
			fetchRecommendationsStageSummary(operatorLogin),
		]);

		if (!detailedResult.ok) {
			setRecommendationRows([]);
			setRecommendationsError(detailedResult.error);
		} else {
			setRecommendationRows(detailedResult.data.rows);
		}

		if (!summaryResult.ok) {
			setRecommendationSummary(null);
			setRecommendationSummaryError(summaryResult.error);
		} else {
			setRecommendationSummary(summaryResult.data);
		}

		setIsRecommendationsLoading(false);
		setIsRecommendationSummaryLoading(false);
	}, [operatorLogin]);

	useEffect(() => {
		void loadInspections();
	}, [loadInspections]);

	useEffect(() => {
		void loadRecommendations();
	}, [loadRecommendations]);

	useEffect(() => {
		if (isLoading || rows.length === 0) {
			return;
		}

		const leaderCurrentCount = rows.filter((row) => row.isLeaderCurrentUser).length;
		const leaderInManagerTeamCount = rows.filter(
			(row) => row.isLeaderInManagerTeam,
		).length;
		const memberCurrentCount = rows.filter((row) => row.isMemberCurrentUser).length;
		const memberInManagerTeamCount = rows.filter(
			(row) => row.isMemberInManagerTeam,
		).length;

		console.groupCollapsed("[Dashboard][inspections-detailed] flags summary");
		console.info("role", authRole);
		console.info("rows", rows.length);
		console.info("isLeaderCurrentUser", leaderCurrentCount);
		console.info("isLeaderInManagerTeam", leaderInManagerTeamCount);
		console.info("isMemberCurrentUser", memberCurrentCount);
		console.info("isMemberInManagerTeam", memberInManagerTeamCount);
		console.table(
			rows.slice(0, 15).map((row) => ({
				kodInspekcji: row.kodInspekcji,
				status: row.statusInspekcji,
				inspektorKierujacy: row.inspektorKierujacy,
				isLeaderCurrentUser: row.isLeaderCurrentUser,
				isLeaderInManagerTeam: row.isLeaderInManagerTeam,
				isMemberCurrentUser: row.isMemberCurrentUser,
				isMemberInManagerTeam: row.isMemberInManagerTeam,
			})),
		);
		console.groupEnd();
	}, [authRole, isLoading, rows]);

	const orderedGroups = useMemo(
		() =>
			(stageSummary?.groups ?? [])
				.filter((group) => {
					const normalizedCode = group.stageGroupCode.trim().toLowerCase();
					const normalizedLabel = group.stageGroupLabel.trim().toLowerCase();
					return normalizedCode !== "unassigned" && normalizedLabel !== "nieprzypisane";
				})
				.slice()
				.sort((left, right) => left.stageGroupOrder - right.stageGroupOrder)
				.map((group) => ({
					...group,
					stageGroupLabel: normalizePolishLabel(group.stageGroupLabel),
					subgroups: group.subgroups
						.slice()
						.sort(
							(leftSubgroup, rightSubgroup) =>
								leftSubgroup.stageSubgroupOrder - rightSubgroup.stageSubgroupOrder,
						)
						.map((subgroup) => ({
							...subgroup,
							stageSubgroupLabel: normalizePolishLabel(subgroup.stageSubgroupLabel),
						})),
				})),
		[stageSummary],
	);

	const stageOverviewData = useMemo<StageOverviewSlice[]>(
		() =>
			orderedGroups
				.filter(
					(group) =>
						!isInformationalClosedGroup(group.stageGroupCode, group.stageGroupLabel),
				)
				.map((group) => ({
					stageGroupCode: group.stageGroupCode,
					stageGroupLabel: group.stageGroupLabel,
					count: group.count,
				})),
		[orderedGroups],
	);

	const visibleStageGroups = useMemo(
		() =>
			orderedGroups.filter(
				(group) =>
					!isInformationalClosedGroup(group.stageGroupCode, group.stageGroupLabel),
			),
		[orderedGroups],
	);

	const orderedRecommendationGroups = useMemo(
		() =>
			(recommendationSummary?.groups ?? [])
				.slice()
				.sort((left, right) => left.stageGroupOrder - right.stageGroupOrder)
				.map((group) => ({
					...group,
					stageGroupLabel: normalizePolishLabel(group.stageGroupLabel),
				})),
		[recommendationSummary],
	);

	const recommendationOverviewData = useMemo<StageOverviewSlice[]>(
		() =>
			orderedRecommendationGroups.map((group) => ({
				stageGroupCode: group.stageGroupCode,
				stageGroupLabel: group.stageGroupLabel,
				count: group.count,
			})),
		[orderedRecommendationGroups],
	);

	const filteredRecommendationRows = useMemo(() => {
		if (selectedRecommendationStatuses.length === 0) {
			return recommendationRows;
		}

		const selectedCodes = new Set(
			selectedRecommendationStatuses.map((filter) => filter.stageGroupCode.trim().toLowerCase()),
		);
		const selectedLabels = new Set(
			selectedRecommendationStatuses.map((filter) =>
				filter.stageGroupLabel.trim().toLowerCase(),
			),
		);

		return recommendationRows.filter((row) => {
			const normalizedStatus = row.status.trim().toLowerCase();
			return selectedCodes.has(normalizedStatus) || selectedLabels.has(normalizedStatus);
		});
	}, [recommendationRows, selectedRecommendationStatuses]);

	const filteredRows = useMemo(() => {
		if (selectedStageFilters.length === 0) {
			return rows;
		}

		const selectedSubgroupCodes = new Set(
			selectedStageFilters.map((filter) => filter.stageSubgroupCode),
		);
		const selectedGroupCodes = new Set(
			selectedStageFilters.map((filter) => filter.stageGroupCode),
		);

		return rows.filter((row) => {
			const hasSubgroupCode = row.stageSubgroupCode.trim().length > 0;
			if (hasSubgroupCode) {
				return selectedSubgroupCodes.has(row.stageSubgroupCode);
			}

			return selectedGroupCodes.has(row.stageGroupCode);
		});
	}, [rows, selectedStageFilters]);

	const startColumnResize = useCallback(
		(columnKey: keyof ReportInspectionDetailedRow, event: React.MouseEvent) => {
			event.preventDefault();
			event.stopPropagation();

			const startX = event.clientX;
			const startWidth = columnWidths[columnKey] ?? 180;
			const minWidth = TABLE_COLUMNS.find((column) => column.key === columnKey)?.minWidth ?? 100;

			const handleMouseMove = (mouseEvent: MouseEvent) => {
				const deltaX = mouseEvent.clientX - startX;
				setColumnWidths((current) => ({
					...current,
					[columnKey]: Math.max(minWidth, startWidth + deltaX),
				}));
			};

			const handleMouseUp = () => {
				window.removeEventListener("mousemove", handleMouseMove);
				window.removeEventListener("mouseup", handleMouseUp);
				document.body.style.cursor = "";
				document.body.style.userSelect = "";
			};

			document.body.style.cursor = "col-resize";
			document.body.style.userSelect = "none";
			window.addEventListener("mousemove", handleMouseMove);
			window.addEventListener("mouseup", handleMouseUp);
		},
		[columnWidths],
	);

	const startRecommendationColumnResize = useCallback(
		(columnKey: keyof ReportRecommendationDetailedRow, event: React.MouseEvent) => {
			event.preventDefault();
			event.stopPropagation();

			const startX = event.clientX;
			const startWidth = recommendationColumnWidths[columnKey] ?? 160;
			const minWidth =
				RECOMMENDATION_TABLE_COLUMNS.find((column) => column.key === columnKey)?.minWidth ?? 100;

			const handleMouseMove = (mouseEvent: MouseEvent) => {
				const deltaX = mouseEvent.clientX - startX;
				setRecommendationColumnWidths((current) => ({
					...current,
					[columnKey]: Math.max(minWidth, startWidth + deltaX),
				}));
			};

			const handleMouseUp = () => {
				window.removeEventListener("mousemove", handleMouseMove);
				window.removeEventListener("mouseup", handleMouseUp);
				document.body.style.cursor = "";
				document.body.style.userSelect = "";
			};

			document.body.style.cursor = "col-resize";
			document.body.style.userSelect = "none";
			window.addEventListener("mousemove", handleMouseMove);
			window.addEventListener("mouseup", handleMouseUp);
		},
		[recommendationColumnWidths],
	);

	return (
		<section className="flex h-full min-h-0 w-full flex-col py-2">
			{error ? (
				<p className="mb-3 rounded-lg border border-rose-300/50 bg-rose-950/30 px-3 py-2 text-rose-100 text-sm">
					{error}
				</p>
			) : null}

			<div className="mb-3 inline-flex w-fit rounded-lg border border-slate-200 bg-white p-1 shadow-sm">
				<button
					type="button"
					onClick={() => setActiveTopSection("inspections")}
					className={`rounded-md px-3 py-1.5 font-medium text-xs transition-colors ${
						activeTopSection === "inspections"
							? "bg-[#1f4f8f] text-white"
							: "text-slate-600 hover:bg-slate-100 hover:text-slate-800"
					}`}
				>
					Inspekcje
				</button>
				<button
					type="button"
					onClick={() => setActiveTopSection("recommendations")}
					className={`rounded-md px-3 py-1.5 font-medium text-xs transition-colors ${
						activeTopSection === "recommendations"
							? "bg-[#1f4f8f] text-white"
							: "text-slate-600 hover:bg-slate-100 hover:text-slate-800"
					}`}
				>
					Zalecenia
				</button>
			</div>

			{activeTopSection === "inspections" ? (
				<>
			<div className="mb-2 shrink-0 grid grid-cols-1 gap-3 xl:grid-cols-2">
				<div className="flex h-full flex-col rounded-xl border border-slate-200 bg-white p-3 text-slate-900 shadow-[0_12px_30px_rgba(2,8,23,0.1)]">
					<div className="mb-2">
						<h3 className="font-semibold text-[15px]">Podsumowanie</h3>
					</div>

					{isStageSummaryLoading ? (
						<div className="flex h-full min-h-[260px] items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-slate-500 text-sm">
							Ładowanie wykresu etapów...
						</div>
					) : stageOverviewData.length > 0 ? (
						<div className="flex h-full min-h-[260px] flex-1 rounded-lg border border-slate-200 bg-white p-2.5">
							<div className="grid h-full w-full grid-cols-1 items-center gap-2.5 lg:grid-cols-[minmax(0,1fr)_220px]">
								<div className="mx-auto h-[190px] w-full max-w-[400px] sm:h-[220px] lg:h-full lg:min-h-[220px]">
								<ResponsiveContainer width="100%" height="100%">
									<PieChart margin={{ top: 16, right: 16, bottom: 16, left: 16 }}>
										<Pie
											data={stageOverviewData}
											dataKey="count"
											nameKey="stageGroupLabel"
											cx="50%"
											cy="50%"
											innerRadius="55%"
											outerRadius="84%"
											paddingAngle={2}
											labelLine={false}
											label={({
												cx,
												cy,
												midAngle,
												innerRadius,
												outerRadius,
												value,
											}) => {
												if (
													typeof value !== "number" ||
													value <= 0 ||
													typeof cx !== "number" ||
													typeof cy !== "number" ||
													typeof midAngle !== "number" ||
													typeof innerRadius !== "number" ||
													typeof outerRadius !== "number"
												) {
													return null;
												}

												const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
												const angle = (-midAngle * Math.PI) / 180;
												const x = cx + radius * Math.cos(angle);
												const y = cy + radius * Math.sin(angle);

												return (
													<text
														x={x}
														y={y}
														fill="#ffffff"
														fontSize={16}
														fontWeight={700}
														textAnchor="middle"
														dominantBaseline="central"
													>
														{value}
													</text>
												);
											}}
										>
											{stageOverviewData.map((entry, index) => (
												<Cell
													key={entry.stageGroupCode}
													fill={
														STAGE_OVERVIEW_COLORS[index % STAGE_OVERVIEW_COLORS.length]
													}
												/>
											))}
										</Pie>
										<Tooltip
											formatter={(value) => [`${value}`, "Liczba"]}
											contentStyle={{ borderRadius: 8, borderColor: "#cbd5e1" }}
										/>
									</PieChart>
								</ResponsiveContainer>
							</div>
								<ul className="subtle-horizontal-scroll space-y-1.5 overflow-x-auto pr-1 text-slate-700 text-[12px]">
									{stageOverviewData.map((slice, index) => (
										<li key={slice.stageGroupCode} className="flex w-max items-center gap-2 whitespace-nowrap leading-tight">
											<span
												className="inline-block h-3 w-3 shrink-0 rounded-full"
												style={{
													backgroundColor:
														STAGE_OVERVIEW_COLORS[index % STAGE_OVERVIEW_COLORS.length],
												}}
											/>
											<span className="whitespace-nowrap">{`${slice.stageGroupLabel} (${slice.count})`}</span>
										</li>
									))}
								</ul>
							</div>
						</div>
					) : (
						<div className="flex h-full min-h-[260px] items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-slate-500 text-sm">
							Brak danych etapów do wykresu.
						</div>
					)}
				</div>

				<div className="rounded-xl border border-slate-200 bg-white p-3 text-slate-900 shadow-[0_12px_30px_rgba(2,8,23,0.1)]">
				<div className="mb-2 flex flex-wrap items-center justify-between gap-3">
					<div>
						<h3 className="font-semibold text-[15px]">Etapy inspekcji</h3>
					</div>
					<button
						type="button"
						onClick={() => setSelectedStageFilters([])}
						disabled={selectedStageFilters.length === 0}
						className={`px-1 py-1 text-xs transition-colors ${
							selectedStageFilters.length === 0
								? "cursor-not-allowed text-slate-400"
								: "cursor-pointer font-semibold text-blue-600 hover:text-blue-700"
						}`}
					>
						Wyczyść filtry
					</button>
				</div>

				{stageSummaryError ? (
					<p className="mb-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-amber-900 text-xs">
						{stageSummaryError}
					</p>
				) : null}

				{isStageSummaryLoading ? (
					<div className="flex h-60 items-center justify-center text-slate-500 text-sm">
						Ładowanie wykresu etapów...
					</div>
				) : visibleStageGroups.length === 0 ? (
					<div className="flex h-60 items-center justify-center text-slate-500 text-sm">
						Brak danych etapów do wykresu.
					</div>
				) : (
					<>
						<div className="rounded-lg border border-slate-200 bg-slate-50 p-2">
							{visibleStageGroups.map((group) => (
								<div
									key={group.stageGroupCode}
									className="grid grid-cols-1 gap-2 border-slate-200 border-t py-1 first:border-t-0 md:grid-cols-[180px_minmax(0,1fr)]"
								>
									<div className="pt-1 font-semibold text-[11px] uppercase tracking-wide text-slate-700 md:pr-2">
										{group.stageGroupLabel}
									</div>
									<div className="space-y-1.5">
										{group.subgroups.map((subgroup) => {
											const isSelected =
												selectedStageFilters.some(
													(filter) =>
														filter.stageGroupCode === group.stageGroupCode &&
														filter.stageSubgroupCode === subgroup.stageSubgroupCode,
												);

											return (
												<button
													key={subgroup.stageSubgroupCode}
													type="button"
													onClick={() => {
														setSelectedStageFilters((current) => {
															const isAlreadySelected = current.some(
																(filter) =>
																	filter.stageGroupCode === group.stageGroupCode &&
																	filter.stageSubgroupCode === subgroup.stageSubgroupCode,
															);

															if (isAlreadySelected) {
																return current.filter(
																	(filter) =>
																		!(
																			filter.stageGroupCode === group.stageGroupCode &&
																			filter.stageSubgroupCode === subgroup.stageSubgroupCode
																		),
																);
															}

															return [
																...current,
																{
																	stageGroupCode: group.stageGroupCode,
																	stageGroupLabel: group.stageGroupLabel,
																	stageSubgroupCode: subgroup.stageSubgroupCode,
																	stageSubgroupLabel: subgroup.stageSubgroupLabel,
																},
															];
														});
													}}
													className={`grid w-full grid-cols-[55%_45%] items-center rounded px-1 py-0.5 text-left text-sm transition-colors ${
														isSelected ? "cursor-pointer bg-blue-100" : "cursor-pointer hover:bg-slate-100"
													}`}
												>
													<span className="pr-3 text-slate-700 text-sm">
														{subgroup.stageSubgroupLabel}
													</span>
													<span className="flex items-center justify-end border-slate-200 border-l pl-3">
														<span className="inline-flex min-w-8 items-center justify-center rounded-md bg-slate-200 px-2 py-0.5 font-semibold text-slate-800 text-xs">
															{subgroup.count}
														</span>
													</span>
												</button>
											);
										})}
									</div>
								</div>
							))}
						</div>
					</>
				)}
				</div>
			</div>

			<div className="min-h-0 flex-1">
			<TableSurface
				isLoading={isLoading}
				errorMessage={error}
				containerClassName="h-full"
				scrollAreaClassName="h-full min-h-0 [scrollbar-gutter:stable]"
			>
				<table className="w-full min-w-max border-collapse font-sans text-slate-900 text-sm">
					<thead>
						<tr className="bg-slate-100 text-slate-800">
							{TABLE_COLUMNS.map((column) => (
								<th
									key={String(column.key)}
									className="sticky top-0 z-10 border-slate-300 border-b bg-slate-100 px-3 py-2 text-left font-semibold"
									style={{ width: columnWidths[column.key], minWidth: column.minWidth }}
								>
									<span className="block truncate pr-3">{column.label}</span>
									<button
										type="button"
										onMouseDown={(event) => startColumnResize(column.key, event)}
										className="absolute top-0 right-0 h-full w-2 cursor-col-resize border-l border-slate-300/80 bg-transparent hover:bg-slate-300/40"
										aria-label={`Zmień szerokość kolumny ${column.label}`}
										title="Przeciągnij, aby zmienić szerokość kolumny"
									/>
								</th>
							))}
						</tr>
					</thead>
					<tbody>
						{filteredRows.length === 0 ? (
							<tr>
								<td colSpan={TABLE_COLUMNS.length} className="px-3 py-8 text-center text-slate-500 text-sm">
									{selectedStageFilters.length > 0
										? "Brak danych dla wybranego segmentu wykresu."
										: "Brak danych do wyświetlenia."}
								</td>
							</tr>
						) : null}

						{filteredRows.map((row, index) => (
									(() => {
										const shouldHighlightRow =
											authRole === "inspector"
												? row.isLeaderCurrentUser
												: authRole === "team_lead"
													? row.isLeaderInManagerTeam
													: false;

										return (
									<tr
										key={`${row.kodInspekcji}-${index}`}
										className={`border-slate-200 border-b transition-colors last:border-b-0 ${
											shouldHighlightRow
												? "bg-[#eef5ff] hover:bg-[#e3efff]"
												: "bg-white hover:bg-slate-50"
										}`}
									>
										{TABLE_COLUMNS.map((column, columnIndex) => (
											<td
												key={`${row.kodInspekcji}-${index}-${String(column.key)}`}
												className={`px-3 py-2.5 align-top ${
													shouldHighlightRow
														? columnIndex === 0
															? "border-slate-200 border-y-2 border-l-4 bg-[#f7faff]"
															: columnIndex === TABLE_COLUMNS.length - 1
																? "border-slate-200 border-y-2 border-r-2 bg-[#f7faff]"
																: "border-slate-200 border-y-2 bg-[#f7faff]"
														: ""
												}`}
												style={{ width: columnWidths[column.key], minWidth: column.minWidth }}
											>
												{column.key === "zakresInspekcji" ? (
													(() => {
														const rawValue = String(row[column.key] ?? "");
														const scopeItems = rawValue
															.split(/\r?\n|;/)
															.map((item) => item.trim())
															.filter((item) => item && item !== "-");

														if (scopeItems.length === 0) {
															return "-";
														}

														return (
															<div className={scopeItems.length > 5 ? "subtle-vertical-scroll max-h-28 space-y-1 overflow-y-auto pr-1" : "space-y-1"}>
																{scopeItems.map((item, itemIndex) => (
																	<div key={`${row.kodInspekcji}-${index}-scope-${itemIndex}`} className="whitespace-normal break-words">
																		{item}
																	</div>
																))}
															</div>
														);
													})()
												) : column.key === "statusInspekcji" ? (
													(() => {
														const isWnType = row.inspekcja === "W";
														const level = isWnType
															? row.wartoscLiczbowaPrzedzialuAlt
															: row.wartoscLiczbowaPrzedzialu;
														const daysSinceEnd = row.liczbaDniOdKoncaInspekcjiDoDzis;
														const showIcon = level === 1 || level === 2 || level === 3;
														const documentLabel = isWnType ? "sprawozdania" : "protokołu";
														const iconClassName =
															level === 1
																? "text-yellow-500"
																: level === 2
																	? "text-orange-500"
																	: "text-red-600";
														const tooltipText =
															level === 1 || level === 2
																? typeof daysSinceEnd === "number"
																	? `Pozostało ${daysSinceEnd} dni na napisanie ${documentLabel}.`
																	: `Pozostało - dni na napisanie ${documentLabel}.`
																: typeof daysSinceEnd === "number"
																	? daysSinceEnd === 30
																		? `Dziś mija termin oddania ${documentLabel}`
																		: daysSinceEnd > 30
																			? `Jest ${daysSinceEnd - 30} dni po terminie oddania ${documentLabel}`
																			: `Pozostało ${daysSinceEnd} dni na napisanie ${documentLabel}.`
																	: `Brak danych o terminie oddania ${documentLabel}.`;

														return (
															<div className="flex items-start justify-between gap-2">
																<span className="whitespace-normal break-words">{String(row[column.key] ?? "-")}</span>
																{showIcon ? (
																	<span title={tooltipText} aria-label={tooltipText}>
																		<AlertTriangle className={`mt-0.5 h-4 w-4 shrink-0 ${iconClassName}`} />
																	</span>
																) : null}
															</div>
														);
													})()
												) : column.key === "inspekcja" ? (
													row.inspekcja === "W" ? "WN" : row.inspekcja
												) : column.key === "kodInspekcji" ? (
													<button
														type="button"
														onClick={() => {
															const inspectionCode = String(row.kodInspekcji ?? "").trim();
															if (!inspectionCode || typeof window === "undefined") {
																return;
															}

															window.sessionStorage.setItem(
																DASHBOARD_OPEN_INSPECTION_CODE_KEY,
																inspectionCode,
															);
															window.dispatchEvent(
																new CustomEvent(DASHBOARD_OPEN_INSPECTION_EVENT, {
																	detail: { inspectionCode },
																}),
															);
														}}
														className="cursor-pointer rounded px-1 text-left text-[#1f4f8f] underline decoration-[#9bb8de] underline-offset-2 transition-colors hover:text-[#163a68]"
														title="Przejdź do rejestru Inspekcje i zaznacz ten rekord"
													>
														{String(row.kodInspekcji ?? "-")}
													</button>
												) : (
													String(row[column.key] ?? "-")
												)}
											</td>
										))}
									</tr>
										);
									})()
							  ))}
					</tbody>
				</table>
			</TableSurface>
			</div>
				</>
			) : (
				<>
					<div className="mb-2 shrink-0 grid grid-cols-1 gap-3 xl:grid-cols-2">
						<div className="flex h-full flex-col rounded-xl border border-slate-200 bg-white p-3 text-slate-900 shadow-[0_12px_30px_rgba(2,8,23,0.1)]">
							<div className="mb-2">
								<h3 className="font-semibold text-[15px]">Podsumowanie zaleceń</h3>
							</div>

							{isRecommendationSummaryLoading ? (
								<div className="flex h-full min-h-[260px] items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-slate-500 text-sm">
									Ładowanie wykresu statusów...
								</div>
							) : recommendationOverviewData.length > 0 ? (
								<div className="flex h-full min-h-[260px] flex-1 rounded-lg border border-slate-200 bg-white p-2.5">
									<div className="-ml-4 grid h-full w-full grid-cols-1 items-center gap-2.5 lg:grid-cols-[minmax(0,1fr)_300px]">
										<div className="ml-0 mr-auto h-[190px] w-full max-w-[400px] sm:h-[220px] lg:h-full lg:min-h-[220px]">
											<ResponsiveContainer width="100%" height="100%">
												<PieChart margin={{ top: 16, right: 16, bottom: 16, left: 16 }}>
													<Pie
														data={recommendationOverviewData}
														dataKey="count"
														nameKey="stageGroupLabel"
														cx="50%"
														cy="50%"
														innerRadius="55%"
														outerRadius="84%"
														paddingAngle={2}
														labelLine={false}
														label={({
															cx,
															cy,
															midAngle,
															innerRadius,
															outerRadius,
															value,
														}) => {
															if (
																typeof value !== "number" ||
																value <= 0 ||
																typeof cx !== "number" ||
																typeof cy !== "number" ||
																typeof midAngle !== "number" ||
																typeof innerRadius !== "number" ||
																typeof outerRadius !== "number"
															) {
																return null;
															}

															const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
															const angle = (-midAngle * Math.PI) / 180;
															const x = cx + radius * Math.cos(angle);
															const y = cy + radius * Math.sin(angle);

															return (
																<text
																	x={x}
																	y={y}
																	fill="#ffffff"
																	fontSize={16}
																	fontWeight={700}
																	textAnchor="middle"
																	dominantBaseline="central"
																>
																	{value}
																</text>
															);
														}}
													>
														{recommendationOverviewData.map((entry, index) => (
															<Cell
																key={entry.stageGroupCode}
																fill={STAGE_OVERVIEW_COLORS[index % STAGE_OVERVIEW_COLORS.length]}
															/>
														))}
													</Pie>
													<Tooltip
														formatter={(value) => [`${value}`, "Liczba"]}
														contentStyle={{ borderRadius: 8, borderColor: "#cbd5e1" }}
													/>
												</PieChart>
											</ResponsiveContainer>
										</div>
										<ul className="-ml-2 space-y-1.5 pr-1 text-slate-700 text-[12px]">
											{recommendationOverviewData.map((slice, index) => (
												<li key={slice.stageGroupCode} className="flex items-center gap-2 whitespace-nowrap leading-tight">
													<span
														className="inline-block h-3 w-3 shrink-0 rounded-full"
														style={{
															backgroundColor:
																STAGE_OVERVIEW_COLORS[index % STAGE_OVERVIEW_COLORS.length],
														}}
													/>
													<span className="whitespace-nowrap">{`${slice.stageGroupLabel} (${slice.count})`}</span>
												</li>
											))}
										</ul>
									</div>
								</div>
							) : (
								<div className="flex h-full min-h-[260px] items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-slate-500 text-sm">
									Brak danych statusów zaleceń.
								</div>
							)}
						</div>

						<div className="rounded-xl border border-slate-200 bg-white p-3 text-slate-900 shadow-[0_12px_30px_rgba(2,8,23,0.1)]">
							<div className="mb-2 flex flex-wrap items-center justify-between gap-3">
								<div>
									<h3 className="font-semibold text-[15px]">Statusy zaleceń</h3>
								</div>
								<button
									type="button"
									onClick={() => setSelectedRecommendationStatuses([])}
									disabled={selectedRecommendationStatuses.length === 0}
									className={`px-1 py-1 text-xs transition-colors ${
										selectedRecommendationStatuses.length === 0
											? "cursor-not-allowed text-slate-400"
											: "cursor-pointer font-semibold text-blue-600 hover:text-blue-700"
									}`}
								>
									Wyczyść filtry
								</button>
							</div>

							{recommendationSummaryError ? (
								<p className="mb-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-amber-900 text-xs">
									{recommendationSummaryError}
								</p>
							) : null}

							{isRecommendationSummaryLoading ? (
								<div className="flex h-60 items-center justify-center text-slate-500 text-sm">
									Ładowanie statusów zaleceń...
								</div>
							) : orderedRecommendationGroups.length === 0 ? (
								<div className="flex h-60 items-center justify-center text-slate-500 text-sm">
									Brak statusów zaleceń.
								</div>
							) : (
								<div className="rounded-lg border border-slate-200 bg-slate-50 p-2">
									<div className="space-y-1.5">
										{orderedRecommendationGroups.map((group) => {
											const isSelected = selectedRecommendationStatuses.some(
												(filter) => filter.stageGroupCode === group.stageGroupCode,
											);

											return (
												<button
													key={group.stageGroupCode}
													type="button"
													onClick={() => {
														setSelectedRecommendationStatuses((current) => {
															const alreadySelected = current.some(
																(item) => item.stageGroupCode === group.stageGroupCode,
															);
															if (alreadySelected) {
																return current.filter(
																	(item) => item.stageGroupCode !== group.stageGroupCode,
																);
															}

															return [
																...current,
																{
																	stageGroupCode: group.stageGroupCode,
																	stageGroupLabel: group.stageGroupLabel,
																},
															];
														});
													}}
													className={`grid w-full grid-cols-[minmax(0,1fr)_auto] items-center rounded px-2 py-1 text-left text-sm transition-colors ${
														isSelected ? "cursor-pointer bg-blue-100" : "cursor-pointer hover:bg-slate-100"
													}`}
												>
													<span className="truncate pr-3 text-slate-700 text-sm">
														{group.stageGroupLabel}
													</span>
													<span className="inline-flex min-w-8 items-center justify-center rounded-md bg-slate-200 px-2 py-0.5 font-semibold text-slate-800 text-xs">
														{group.count}
													</span>
												</button>
											);
										})}
									</div>
								</div>
							)}
						</div>
					</div>

					<div className="min-h-0 flex-1">
						<TableSurface
							isLoading={isRecommendationsLoading}
							errorMessage={recommendationsError}
							containerClassName="h-full"
							scrollAreaClassName="h-full min-h-0 [scrollbar-gutter:stable]"
						>
							<table className="w-full min-w-max border-collapse font-sans text-slate-900 text-sm">
								<thead>
									<tr className="bg-slate-100 text-slate-800">
										{RECOMMENDATION_TABLE_COLUMNS.map((column) => (
											<th
												key={String(column.key)}
												className="sticky top-0 z-10 border-slate-300 border-b bg-slate-100 px-3 py-2 text-left font-semibold"
												style={{
													width: recommendationColumnWidths[column.key],
													minWidth: column.minWidth,
												}}
											>
												<span className="block truncate pr-3">{column.label}</span>
												<button
													type="button"
													onMouseDown={(event) =>
														startRecommendationColumnResize(column.key, event)
													}
													className="absolute top-0 right-0 h-full w-2 cursor-col-resize border-l border-slate-300/80 bg-transparent hover:bg-slate-300/40"
													aria-label={`Zmień szerokość kolumny ${column.label}`}
													title="Przeciągnij, aby zmienić szerokość kolumny"
												/>
											</th>
										))}
									</tr>
								</thead>
								<tbody>
									{filteredRecommendationRows.length === 0 ? (
										<tr>
											<td
												colSpan={RECOMMENDATION_TABLE_COLUMNS.length}
												className="px-3 py-8 text-center text-slate-500 text-sm"
											>
												{selectedRecommendationStatuses.length > 0
													? "Brak danych dla wybranego statusu zaleceń."
													: "Brak danych do wyświetlenia."}
											</td>
										</tr>
									) : null}

									{filteredRecommendationRows.map((row, index) => (
										<tr
											key={`${row.recommendationId}-${row.inspectionId}-${index}`}
											className="border-slate-200 border-b bg-white transition-colors last:border-b-0 hover:bg-slate-50"
										>
											{RECOMMENDATION_TABLE_COLUMNS.map((column) => (
												<td
													key={`${row.recommendationId}-${row.inspectionId}-${index}-${String(column.key)}`}
													className="px-3 py-2.5 align-top"
													style={{
														width: recommendationColumnWidths[column.key],
														minWidth: column.minWidth,
													}}
												>
													{column.key === "terminWykonaniaZalecen" ? (
														(() => {
															const value = String(row[column.key] ?? "-");
															const terms = value
																.split(",")
																.map((item) => item.trim())
																.filter(Boolean);

															if (terms.length <= 1) {
																return value;
															}

															return (
																<div className="space-y-1">
																	{terms.map((term, termIndex) => (
																		<div key={`${row.recommendationId}-${termIndex}`}>{term}</div>
																	))}
																</div>
															);
														})()
													) : column.key === "inspectionId" ? (
														(() => {
															const inspectionCode = String(row.inspectionId ?? "").trim();
															if (!inspectionCode || inspectionCode === "-") {
																return "-";
															}

															return (
																<button
																	type="button"
																	onClick={() => {
																		if (typeof window === "undefined") {
																			return;
																		}

																		window.sessionStorage.setItem(
																			DASHBOARD_OPEN_INSPECTION_CODE_KEY,
																			inspectionCode,
																		);
																		window.dispatchEvent(
																			new CustomEvent(DASHBOARD_OPEN_INSPECTION_EVENT, {
																				detail: { inspectionCode },
																			}),
																		);
																	}}
																	className="cursor-pointer rounded px-1 text-left text-[#1f4f8f] underline decoration-[#9bb8de] underline-offset-2 transition-colors hover:text-[#163a68]"
																	title="Przejdź do rejestru Inspekcje i zaznacz ten rekord"
																>
																	{inspectionCode}
																</button>
															);
														})()
													) : column.key === "recommendationId" ? (
														(() => {
															const recommendationCode = String(row.recommendationId ?? "").trim();
															if (!recommendationCode || recommendationCode === "-") {
																return "-";
															}

															return (
																<button
																	type="button"
																	onClick={() => {
																		if (typeof window === "undefined") {
																			return;
																		}

																		window.sessionStorage.setItem(
																			DASHBOARD_OPEN_RECOMMENDATION_CODE_KEY,
																			recommendationCode,
																		);
																		window.dispatchEvent(
																			new CustomEvent(DASHBOARD_OPEN_RECOMMENDATION_EVENT, {
																				detail: { recommendationCode },
																			}),
																		);
																	}}
																	className="cursor-pointer rounded px-1 text-left text-[#1f4f8f] underline decoration-[#9bb8de] underline-offset-2 transition-colors hover:text-[#163a68]"
																	title="Przejdź do rejestru Zalecenia i zaznacz ten rekord"
																>
																	{recommendationCode}
																</button>
															);
														})()
													) : (
														String(row[column.key] ?? "-")
													)}
												</td>
											))}
										</tr>
									))}
								</tbody>
							</table>
						</TableSurface>
					</div>
				</>
			)}
		</section>
	);
}
