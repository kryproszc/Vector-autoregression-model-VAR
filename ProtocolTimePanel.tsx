"use client";

import {
	CartesianGrid,
	ComposedChart,
	Line,
	ReferenceArea,
	ResponsiveContainer,
	Scatter,
	XAxis,
	YAxis,
} from "recharts";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

import { getStoredAuthSession } from "@/features/auth/session";
import { normalizeAuthRole } from "@/features/auth/types";
import { fetchInspectionsTimeAnalytics } from "@/features/reports/api";
import type {
	ReportInspectionDetailedRow,
	TimeReportOverallColumn,
	TimeReportOverallRow,
	TimeReportScatterRow,
	TimeReportSummaryPivotRow,
	TimeReportYearCountByTeamRow,
	TimeReportYearCountRow,
	TimeReportTrendMode,
	TimeReportTrendRow,
} from "@/features/reports/types";

type TimeReportPanelProps = {
	operatorLogin: string;
	title: string;
	inspectionType: "K" | "W";
};

type TrendMode = TimeReportTrendMode;
type TimeRibbonTab = "inspection-data" | "time-summary" | "visualization";

type ChartPointPayload = {
	year: number;
	time: number;
	nazwaPodmiotu: string;
	kontrola: string;
	osobaKierujaca: string;
	zespol: string;
};

type ChartClickTooltipState =
	| {
			kind: "point";
			x: number;
			y: number;
			data: {
				year: number;
				time: number;
				nazwaPodmiotu: string;
				kontrola: string;
				osobaKierujaca: string;
				zespol: string;
			};
	  }
	| {
			kind: "trend";
			x: number;
			y: number;
			year: number;
			trend: number | null;
	  }
	| null;

const INSPECTIONS_CHANGED_EVENT = "inspections:changed";
const YEAR_BOX_HALF_WIDTH = 0.26;
const CHART_SCROLL_THRESHOLD_YEARS = 6;
const CHART_YEAR_SLOT_WIDTH_PX = 260;
const HISTOGRAM_BINS_PER_YEAR = 8;
const HISTOGRAM_INSET_X = 0.012;
const HISTOGRAM_WIDTH_SHARE = 0.75;
function TableLoadingOverlay() {
	return (
		<div className="absolute inset-0 z-20 flex items-center justify-center bg-white/70 backdrop-blur-[1px]">
			<div className="flex items-center gap-3 rounded-full border border-slate-200 bg-white/90 px-4 py-2 shadow-[0_8px_20px_rgba(2,8,23,0.15)]">
				<span className="h-4 w-4 animate-spin rounded-full border-2 border-[#7aa5dc] border-t-[#255087]" />
				<span className="font-medium text-slate-700 text-sm">Ładowanie danych...</span>
			</div>
		</div>
	);
}

const DETAIL_COLUMNS: Array<{
	key: keyof ReportInspectionDetailedRow;
	label: string;
	width: string;
}> = [
	{ key: "nazwaPodmiotu", label: "Nazwa podmiotu", width: "min-w-56" },
	{ key: "kontrola", label: "Kontrola", width: "min-w-40" },
	{ key: "rokPoczatku", label: "Rok", width: "min-w-28" },
	{ key: "osobaKierujaca", label: "Osoba kierująca", width: "min-w-44" },
	{ key: "zespol", label: "Zespół", width: "min-w-24" },
	{ key: "czas", label: "Czas (dni)", width: "min-w-20" },
];

export function TimeReportPanel({
	operatorLogin,
	title,
	inspectionType,
}: TimeReportPanelProps) {
	const [detailRows, setDetailRows] = useState<ReportInspectionDetailedRow[]>([]);
	const [summaryPivotYears, setSummaryPivotYears] = useState<string[]>([]);
	const [summaryPivotRows, setSummaryPivotRows] = useState<TimeReportSummaryPivotRow[]>([]);
	const [yearCountColumns, setYearCountColumns] = useState<string[]>([]);
	const [yearCountRows, setYearCountRows] = useState<TimeReportYearCountRow[]>([]);
	const [yearCountByTeamColumns, setYearCountByTeamColumns] = useState<string[]>([]);
	const [yearCountByTeamRows, setYearCountByTeamRows] = useState<TimeReportYearCountByTeamRow[]>([]);
	const [overallColumns, setOverallColumns] = useState<TimeReportOverallColumn[]>([]);
	const [overallRows, setOverallRows] = useState<TimeReportOverallRow[]>([]);
	const [selectedMetricLabel, setSelectedMetricLabel] = useState("Średni czas");
	const [trendRows, setTrendRows] = useState<TimeReportTrendRow[]>([]);
	const [scatterRows, setScatterRows] = useState<TimeReportScatterRow[]>([]);
	const [leaderDisplayNameByLogin, setLeaderDisplayNameByLogin] = useState<
		Record<string, string>
	>({});
	const [teamOptions, setTeamOptions] = useState<string[]>([]);
	const [startYearOptions, setStartYearOptions] = useState<string[]>([]);
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [trendMode, setTrendMode] = useState<TrendMode>("average");
	const [selectedTeams, setSelectedTeams] = useState<string[]>([]);
	const [selectedStartYears, setSelectedStartYears] = useState<string[]>([]);
	const [teamSearch, setTeamSearch] = useState("");
	const [yearSearch, setYearSearch] = useState("");
	const [activeRibbonTab, setActiveRibbonTab] = useState<TimeRibbonTab>(
		"inspection-data",
	);
	const [showHistogram, setShowHistogram] = useState(false);
	const [showScatterPoints, setShowScatterPoints] = useState(true);
	const [chartClickTooltip, setChartClickTooltip] =
		useState<ChartClickTooltipState>(null);
	const filtersRef = useRef<HTMLDivElement | null>(null);

	const authRole = useMemo(() => {
		const storedRole = getStoredAuthSession()?.user?.rola;
		return normalizeAuthRole(storedRole);
	}, []);

	const isManagerView = authRole === "director" || authRole === "team_lead";

	useEffect(() => {
		let isActive = true;

		const loadPeopleOptions = async () => {
			try {
				const response = await fetch("/api/inspections/people-options", {
					method: "GET",
					headers: {
						"Content-Type": "application/json",
						"X-Operator-Login": operatorLogin,
					},
					cache: "no-store",
				});

				if (!response.ok) {
					if (isActive) {
						setLeaderDisplayNameByLogin({});
					}
					return;
				}

				const payload = (await response.json()) as unknown;
				const users = Array.isArray(payload)
					? payload
					: Array.isArray((payload as { items?: unknown[] })?.items)
						? (((payload as { items?: unknown[] }).items ?? []) as unknown[])
						: [];

				const next: Record<string, string> = {};
				for (const user of users) {
					const source = (user ?? {}) as {
						login?: unknown;
						displayName?: unknown;
						imie?: unknown;
						nazwisko?: unknown;
					};
					const login =
						typeof source.login === "string" ? source.login.trim().toLowerCase() : "";
					if (!login) {
						continue;
					}

					const firstName = typeof source.imie === "string" ? source.imie.trim() : "";
					const lastName =
						typeof source.nazwisko === "string" ? source.nazwisko.trim() : "";
					const fullName = `${firstName} ${lastName}`.trim();
					const displayName =
						typeof source.displayName === "string" ? source.displayName.trim() : "";

					next[login] = fullName || displayName || login;
				}

				if (isActive) {
					setLeaderDisplayNameByLogin(next);
				}
			} catch {
				if (isActive) {
					setLeaderDisplayNameByLogin({});
				}
			}
		};

		void loadPeopleOptions();

		return () => {
			isActive = false;
		};
	}, [operatorLogin]);

	const resolveLeaderDisplayName = useCallback(
		(value: string) => {
			const normalized = value.trim();
			if (!normalized || normalized === "-") {
				return "-";
			}

			return leaderDisplayNameByLogin[normalized.toLowerCase()] ?? normalized;
		},
		[leaderDisplayNameByLogin],
	);

	const detailRowsWithDisplayNames = useMemo(
		() =>
			detailRows.map((row) => ({
				...row,
				osobaKierujaca: resolveLeaderDisplayName(row.osobaKierujaca),
			})),
		[detailRows, resolveLeaderDisplayName],
	);

	const scatterRowsWithDisplayNames = useMemo(
		() =>
			scatterRows.map((row) => ({
				...row,
				osobaKierujaca: resolveLeaderDisplayName(row.osobaKierujaca),
			})),
		[resolveLeaderDisplayName, scatterRows],
	);

	const loadReport = useCallback(async () => {
		setIsLoading(true);
		setError(null);

		const result = await fetchInspectionsTimeAnalytics(operatorLogin, {
			inspectionType,
			trendMode,
			teams: selectedTeams,
			years: selectedStartYears,
		});
		if (!result.ok) {
			setDetailRows([]);
			setSummaryPivotYears([]);
			setSummaryPivotRows([]);
			setYearCountColumns([]);
			setYearCountRows([]);
			setYearCountByTeamColumns([]);
			setYearCountByTeamRows([]);
			setOverallColumns([]);
			setOverallRows([]);
			setSelectedMetricLabel(trendMode === "average" ? "Średni czas" : "Mediana czasu");
			setTrendRows([]);
			setScatterRows([]);
			setTeamOptions([]);
			setStartYearOptions([]);
			setError(result.error);
			setIsLoading(false);
			return;
		}

		setDetailRows(result.data.detailRows);
		setSummaryPivotYears(result.data.summaryPivotYears);
		setSummaryPivotRows(result.data.summaryPivotRows);
		setYearCountColumns(result.data.yearCountColumns);
		setYearCountRows(result.data.yearCountRows);
		setYearCountByTeamColumns(result.data.yearCountByTeamColumns);
		setYearCountByTeamRows(result.data.yearCountByTeamRows);
		setOverallColumns(result.data.overallColumns);
		setOverallRows(result.data.overallRows);
		setSelectedMetricLabel(result.data.selectedMetricLabel);
		setTrendRows(result.data.trendRows);
		setScatterRows(result.data.scatterRows);
		setTeamOptions(result.data.teamOptions);
		setStartYearOptions(result.data.yearOptions);
		setIsLoading(false);
	}, [
		inspectionType,
		operatorLogin,
		selectedStartYears,
		selectedTeams,
		trendMode,
	]);

	useEffect(() => {
		loadReport();
	}, [loadReport]);

	useEffect(() => {
		setChartClickTooltip(null);
	}, [activeRibbonTab, showScatterPoints, trendMode, trendRows, scatterRows]);

	useEffect(() => {
		const handleInspectionsChanged = () => {
			void loadReport();
		};

		window.addEventListener(INSPECTIONS_CHANGED_EVENT, handleInspectionsChanged);
		return () => {
			window.removeEventListener(INSPECTIONS_CHANGED_EVENT, handleInspectionsChanged);
		};
	}, [loadReport]);

	useEffect(() => {
		const handlePointerDown = (event: MouseEvent) => {
			if (!filtersRef.current) {
				return;
			}

			const target = event.target;
			if (!(target instanceof Node)) {
				return;
			}

			if (filtersRef.current.contains(target)) {
				return;
			}

			for (const detailsElement of filtersRef.current.querySelectorAll("details[open]")) {
				(detailsElement as HTMLDetailsElement).open = false;
			}
		};

		document.addEventListener("mousedown", handlePointerDown);
		return () => {
			document.removeEventListener("mousedown", handlePointerDown);
		};
	}, []);

	const selectTrendMode = useCallback(
		(mode: TrendMode, event: React.MouseEvent<HTMLButtonElement>) => {
			setTrendMode(mode);

			const detailsElement = event.currentTarget.closest("details");
			if (detailsElement instanceof HTMLDetailsElement) {
				detailsElement.open = false;
			}
		},
		[],
	);

	const visibleTeamOptions = useMemo(() => {
		const query = teamSearch.trim().toLowerCase();
		if (!query) {
			return teamOptions;
		}

		return teamOptions.filter((team) => team.toLowerCase().includes(query));
	}, [teamOptions, teamSearch]);

	const visibleYearOptions = useMemo(() => {
		const query = yearSearch.trim();
		if (!query) {
			return startYearOptions;
		}

		return startYearOptions.filter((year) => year.includes(query));
	}, [startYearOptions, yearSearch]);

	const toggleTeam = useCallback((team: string) => {
		setSelectedTeams((previous) =>
			previous.includes(team)
				? previous.filter((item) => item !== team)
				: [...previous, team],
		);
	}, []);

	const toggleStartYear = useCallback((year: string) => {
		setSelectedStartYears((previous) =>
			previous.includes(year)
				? previous.filter((item) => item !== year)
				: [...previous, year],
		);
	}, []);

	const clearFilters = useCallback(() => {
		setSelectedTeams([]);
		setSelectedStartYears([]);
		setTeamSearch("");
		setYearSearch("");

		if (!filtersRef.current) {
			return;
		}

		for (const detailsElement of filtersRef.current.querySelectorAll("details[open]")) {
			(detailsElement as HTMLDetailsElement).open = false;
		}
	}, []);

	const hasActiveFilters = useMemo(
		() =>
			selectedTeams.length > 0 ||
			selectedStartYears.length > 0 ||
			teamSearch.trim().length > 0 ||
			yearSearch.trim().length > 0,
		[selectedTeams, selectedStartYears, teamSearch, yearSearch],
	);

	const yearBuckets = useMemo(
		() =>
			Array.from(new Set(trendRows.map((point) => point.year))).sort(
				(left, right) => left - right,
			),
		[trendRows],
	);

	const yearToIndex = useMemo(
		() => new Map(yearBuckets.map((year, index) => [year, index])),
		[yearBuckets],
	);

	const chartPointsData = useMemo(() => {
		const pointsByYear = new Map<number, typeof scatterRowsWithDisplayNames>();

		for (const point of scatterRowsWithDisplayNames) {
			const bucket = pointsByYear.get(point.year) ?? [];
			bucket.push(point);
			pointsByYear.set(point.year, bucket);
		}

		return Array.from(pointsByYear.entries()).flatMap(([year, points]) => {
			const yearIndex = yearToIndex.get(year) ?? 0;
			const innerLeft = yearIndex - YEAR_BOX_HALF_WIDTH + HISTOGRAM_INSET_X;
			const innerRight = yearIndex + YEAR_BOX_HALF_WIDTH - HISTOGRAM_INSET_X;
			const innerWidth = Math.max(0.001, innerRight - innerLeft);
			const histogramWidth = innerWidth * HISTOGRAM_WIDTH_SHARE;
			const pointsWidth = Math.max(0.001, innerWidth - histogramWidth);
			const pointsCenterX = innerLeft + histogramWidth + pointsWidth / 2;

			return points
				.slice()
				.sort((left, right) => left.time - right.time)
				.map((point) => ({
					...point,
					x: pointsCenterX,
				}));
		});
	}, [scatterRowsWithDisplayNames, yearToIndex]);

	const chartTrendData = useMemo(
		() =>
			trendRows.map((row) => ({
				...row,
				x: yearToIndex.get(row.year) ?? 0,
			})),
		[trendRows, yearToIndex],
	);

	const handleChartBackgroundClick = useCallback(() => {
		setChartClickTooltip(null);
	}, []);

	const openPointTooltip = useCallback(
		(point: ChartPointPayload, x: number, y: number) => {
			setChartClickTooltip({
				kind: "point",
				x,
				y,
				data: point,
			});
		},
		[],
	);

	const openTrendTooltip = useCallback(
		(point: { year: number; trend: number | null }, x: number, y: number) => {
			setChartClickTooltip({
				kind: "trend",
				x,
				y,
				year: point.year,
				trend: point.trend,
			});
		},
		[],
	);

	const renderScatterPoint = useCallback(
		(rawProps: unknown) => {
			const props = rawProps as {
				cx?: number;
				cy?: number;
				payload?: ChartPointPayload;
				index?: number;
			};
			const scatterKey = props.payload
				? `scatter-${props.payload.year}-${props.payload.nazwaPodmiotu}-${props.payload.kontrola}-${props.payload.time}`
				: `scatter-empty-${props.index ?? "unknown"}`;
			if (
				typeof props.cx !== "number" ||
				typeof props.cy !== "number" ||
				!props.payload
			) {
				return <g key={scatterKey} />;
			}

			return (
				<circle
					key={scatterKey}
					cx={props.cx}
					cy={props.cy}
					r={4}
					fill="#5d8fc9"
					style={{ cursor: "pointer" }}
					onClick={(event) => {
						event.stopPropagation();
						openPointTooltip(props.payload as ChartPointPayload, props.cx as number, props.cy as number);
					}}
				/>
			);
		},
		[openPointTooltip],
	);

	const renderTrendDot = useCallback(
		(rawProps: unknown) => {
			const props = rawProps as {
				cx?: number;
				cy?: number;
				payload?: { year: number; trend: number | null };
				index?: number;
			};
			const trendKey = props.payload
				? `trend-${props.payload.year}`
				: `trend-empty-${props.index ?? "unknown"}`;
			if (
				typeof props.cx !== "number" ||
				typeof props.cy !== "number" ||
				!props.payload
			) {
				return <g key={trendKey} />;
			}

			if (props.payload.trend === null) {
				return <g key={trendKey} />;
			}

			return (
				<circle
					key={trendKey}
					cx={props.cx}
					cy={props.cy}
					r={4}
					fill="#255087"
					style={{ cursor: "pointer" }}
					onClick={(event) => {
						event.stopPropagation();
						openTrendTooltip(props.payload as { year: number; trend: number | null }, props.cx as number, props.cy as number);
					}}
				/>
			);
		},
		[openTrendTooltip],
	);

	const shouldEnableChartScroll = useMemo(
		() => yearBuckets.length > CHART_SCROLL_THRESHOLD_YEARS,
		[yearBuckets.length],
	);

	const chartScrollableWidthPx = useMemo(
		() => yearBuckets.length * CHART_YEAR_SLOT_WIDTH_PX,
		[yearBuckets.length],
	);

	const xAxisDomain = useMemo<[number, number]>(() => {
		if (yearBuckets.length === 0) {
			return [-0.5, 0.5];
		}

		if (yearBuckets.length === 1) {
			return [-0.6, 0.6];
		}

		return [-0.5, yearBuckets.length - 0.5];
	}, [yearBuckets]);

	const yAxisDomain = useMemo<[number, number]>(() => {
		if (scatterRowsWithDisplayNames.length === 0) {
			return [0, 10];
		}

		const values = scatterRowsWithDisplayNames.map((point) => point.time);
		const minValue = Math.min(...values);
		const maxValue = Math.max(...values);

		if (minValue === maxValue) {
			return [Math.max(0, minValue - 10), maxValue + 10];
		}

		const range = maxValue - minValue;
		const padding = Math.max(6, Math.round(range * 0.12));
		return [Math.max(0, minValue - padding), maxValue + padding];
	}, [scatterRowsWithDisplayNames]);

	const yearRangeAreas = useMemo(() => {
		return chartTrendData.flatMap((yearData) => {
			if (yearData.min === null || yearData.max === null) {
				return [];
			}

			const y1 = Math.min(yearData.min, yearData.max);
			const y2 = Math.max(yearData.min, yearData.max);
			const top = y1 === y2 ? y2 + 1 : y2;

			return {
				key: `range-${yearData.year}`,
				year: yearData.year,
				x: yearData.x,
				y1,
				y2: top,
			};
		});
	}, [chartTrendData]);

	const histogramAreas = useMemo(() => {
		if (yearRangeAreas.length === 0 || scatterRowsWithDisplayNames.length === 0) {
			return [] as Array<{
				key: string;
				x1: number;
				x2: number;
				y1: number;
				y2: number;
				opacity: number;
			}>;
		}

		const pointsByYear = new Map<number, number[]>();

		for (const point of scatterRowsWithDisplayNames) {
			const yearPoints = pointsByYear.get(point.year) ?? [];
			yearPoints.push(point.time);
			pointsByYear.set(point.year, yearPoints);
		}

		const areas: Array<{
			key: string;
			x1: number;
			x2: number;
			y1: number;
			y2: number;
			opacity: number;
		}> = [];

		for (const rangeArea of yearRangeAreas) {
			const times = pointsByYear.get(rangeArea.year) ?? [];
			if (times.length === 0) {
				continue;
			}

			const innerLeft = rangeArea.x - YEAR_BOX_HALF_WIDTH + HISTOGRAM_INSET_X;
			const innerRight = rangeArea.x + YEAR_BOX_HALF_WIDTH - HISTOGRAM_INSET_X;
			const innerWidth = Math.max(0.001, innerRight - innerLeft);
			const histogramRight = innerLeft + innerWidth * HISTOGRAM_WIDTH_SHARE;
			const histogramWidth = Math.max(0.001, histogramRight - innerLeft);
			const valueMin = Math.min(rangeArea.y1, rangeArea.y2);
			const valueMax = Math.max(rangeArea.y1, rangeArea.y2);
			const valueRange = Math.max(1e-6, valueMax - valueMin);
			const counts = Array.from({ length: HISTOGRAM_BINS_PER_YEAR }, () => 0);

			for (const time of times) {
				const clampedTime = Math.min(valueMax - 1e-6, Math.max(valueMin, time));
				const normalized = Math.min(0.999999, Math.max(0, (clampedTime - valueMin) / valueRange));
				const binIndex = Math.floor(normalized * HISTOGRAM_BINS_PER_YEAR);
				counts[binIndex]! += 1;
			}

			const maxCount = Math.max(...counts, 1);

			for (let binIndex = 0; binIndex < HISTOGRAM_BINS_PER_YEAR; binIndex += 1) {
				const count = counts[binIndex] ?? 0;
				if (count === 0) {
					continue;
				}

				const ratio = count / maxCount;
				const barWidth = Math.max(0.004, histogramWidth * ratio);
				const y1 = valueMin + (binIndex / HISTOGRAM_BINS_PER_YEAR) * valueRange;
				const y2 = valueMin + ((binIndex + 1) / HISTOGRAM_BINS_PER_YEAR) * valueRange;

				areas.push({
					key: `hist-${rangeArea.year}-${binIndex}`,
					x1: innerLeft,
					x2: Math.min(histogramRight, innerLeft + barWidth),
					y1,
					y2,
					opacity: 0.14 + ratio * 0.26,
				});
			}
		}

		return areas;
	}, [scatterRowsWithDisplayNames, yearRangeAreas]);

	const formatMetricValue = useCallback((value: number) => {
		return Number.isFinite(value) ? value.toFixed(1) : "0.0";
	}, []);

	const formatSummaryCellValue = useCallback(
		(value: string | number | null) => {
			if (value === null || value === undefined || value === "") {
				return "-";
			}

			if (typeof value === "number") {
				return formatMetricValue(value);
			}

			if (value.trim() !== "") {
				const parsed = Number(value.replace(",", "."));
				if (Number.isFinite(parsed)) {
					return formatMetricValue(parsed);
				}
			}

			return value;
		},
		[formatMetricValue],
	);

	const sortedDetailRows = useMemo(
		() => detailRowsWithDisplayNames,
		[detailRowsWithDisplayNames],
	);
	const shouldScrollSummaryTable = summaryPivotRows.length > 10;
	const metricColumnKey = trendMode === "average" ? "average" : "median";
	const metricColumnLabel = trendMode === "average" ? "Średnia" : "Mediana";
	const documentLabelGenitive = inspectionType === "K" ? "protokołu" : "sprawozdania";
	const volumeLabel = inspectionType === "K" ? "kontroli" : "wizyt nadzorczych";
	const yearlyTimeSectionTitle = `${metricColumnLabel} czasu przygotowania ${documentLabelGenitive} wg roku`;
	const yearlyCountSectionTitle = `Liczba ${volumeLabel} wg roku`;
	const allYearsSectionTitle = "Statystyka i liczba dla wszystkich lat";
	const summaryPivotColumns = useMemo(() => {
		return summaryPivotYears
			.filter((key) => key !== "allYears")
			.map((key) => ({
			key,
			label: key,
		}));
	}, [summaryPivotYears]);

	const resolvedYearCountColumns = useMemo(() => {
		if (yearCountByTeamColumns.length > 0) {
			return yearCountByTeamColumns;
		}

		if (yearCountColumns.length > 0) {
			return yearCountColumns;
		}

		if (summaryPivotColumns.length > 0) {
			return summaryPivotColumns.map((column) => column.key);
		}

		return [] as string[];
	}, [summaryPivotColumns, yearCountByTeamColumns, yearCountColumns]);

	const resolvedYearCountRows = useMemo(() => {
		if (yearCountByTeamRows.length > 0) {
			return yearCountByTeamRows
				.filter((row) => row.inspekcja === inspectionType)
				.map((row) => ({
				label: row.zespol,
				values: row.values,
				}));
		}

		return yearCountRows.map((row) => ({
			label: row.label.split(" - ")[0]?.trim() || row.label,
			values: row.values,
		}));
	}, [inspectionType, yearCountByTeamRows, yearCountRows]);

	const resolvedOverallColumns = useMemo(() => {
		if (overallColumns.length > 0) {
			return overallColumns;
		}

		if (overallRows.length > 0) {
			const firstRow = overallRows[0];
			if (!firstRow) {
				return [];
			}

			const toLabel = (key: string) => {
				if (key === "zespol") return "Zespół";
				if (key === "count") return "Liczba";
				if (key === "average") return "Średnia";
				if (key === "median") return "Mediana";
				return key;
			};

			const toRank = (key: string) => {
				if (key === "zespol") return 0;
				if (key === "median") return 1;
				if (key === "average") return 1;
				if (key === "count") return 2;
				return 10;
			};

			return Object.keys(firstRow)
				.map((key) => ({ key, label: toLabel(key) }))
				.sort((left, right) => toRank(left.key) - toRank(right.key));
		}

		if (summaryPivotRows.length > 0) {
			return [
				{ key: "zespol", label: "Zespół" },
				{ key: metricColumnKey, label: metricColumnLabel },
				{ key: "count", label: "Liczba" },
			];
		}

		return [];
	}, [metricColumnKey, metricColumnLabel, overallColumns, overallRows, summaryPivotRows.length]);

	const resolvedOverallRows = useMemo(() => {
		if (overallRows.length > 0) {
			return overallRows;
		}

		if (summaryPivotRows.length === 0) {
			return [] as TimeReportOverallRow[];
		}

		return summaryPivotRows.map((row) => ({
			zespol: row.zespol,
			[metricColumnKey]: row.values.allYears ?? null,
			count: null,
		}));
	}, [metricColumnKey, overallRows, summaryPivotRows]);

	const hasYearCountData =
		resolvedYearCountColumns.length > 0 && resolvedYearCountRows.length > 0;

	return (
		<section className="flex h-[calc(100vh-4.4rem)] min-h-0 flex-col space-y-4 overflow-hidden rounded-2xl border border-slate-700/70 bg-[#101f39] p-5 shadow-[0_18px_44px_rgba(2,8,23,0.34)]">
			<div className="flex flex-wrap items-center justify-between gap-2">
				<h2 className="font-semibold text-slate-100 text-xl">{title}</h2>
			</div>

			{error ? (
				<p className="rounded-lg border border-rose-300/60 bg-rose-100/90 px-3 py-2 text-rose-800 text-sm">
					{error}
				</p>
			) : null}

			<div ref={filtersRef} className="rounded-lg border border-slate-700/70 bg-[#122c4e] p-3">
				<div className="flex flex-wrap items-center justify-between gap-2">
					<div className="flex flex-wrap items-center gap-2">
						<details className="group relative">
							<summary className="inline-flex w-48 cursor-pointer list-none items-center justify-between rounded-md border border-[#b6c6dc] bg-[#f8fbff] px-3 py-2 font-medium text-slate-800 text-sm">
								<span className="truncate">Zespół: {selectedTeams.length === 0 ? "wszystkie" : selectedTeams.length}</span>
								<ChevronDown size={14} className="shrink-0 text-slate-500 transition-transform duration-200 group-open:rotate-180" />
							</summary>
							<div className="absolute left-0 z-30 mt-2 w-80 rounded-lg border border-[#b6c6dc] bg-[#f8fbff] p-2 shadow-[0_14px_28px_rgba(2,8,23,0.24)]">
								<input
									type="text"
									value={teamSearch}
									onChange={(event) => setTeamSearch(event.target.value)}
									placeholder="Szukaj zespołu"
									className="w-full rounded-md border border-[#b6c6dc] bg-white px-2.5 py-2 text-slate-800 text-sm outline-none placeholder:text-slate-400"
								/>
								<div className="mt-2 max-h-36 space-y-1 overflow-auto pr-1">
									{visibleTeamOptions.map((team) => (
										<label
											key={team}
											className="flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-slate-800 text-sm hover:bg-[#e9f1fc]"
										>
											<input
												type="checkbox"
												checked={selectedTeams.includes(team)}
												onChange={() => toggleTeam(team)}
												className="h-3.5 w-3.5"
											/>
											<span className="truncate">{team}</span>
										</label>
									))}
								</div>
							</div>
						</details>

						<details className="group relative">
							<summary className="inline-flex w-48 cursor-pointer list-none items-center justify-between rounded-md border border-[#b6c6dc] bg-[#f8fbff] px-3 py-2 font-medium text-slate-800 text-sm">
								<span className="truncate">Rok: {selectedStartYears.length === 0 ? "wszystkie" : selectedStartYears.length}</span>
								<ChevronDown size={14} className="shrink-0 text-slate-500 transition-transform duration-200 group-open:rotate-180" />
							</summary>
							<div className="absolute left-0 z-30 mt-2 w-56 rounded-lg border border-[#b6c6dc] bg-[#f8fbff] p-2 shadow-[0_14px_28px_rgba(2,8,23,0.24)]">
								<input
									type="text"
									value={yearSearch}
									onChange={(event) => setYearSearch(event.target.value)}
									placeholder="Szukaj roku"
									className="w-full rounded-md border border-[#b6c6dc] bg-white px-2.5 py-2 text-slate-800 text-sm outline-none placeholder:text-slate-400"
								/>
								<div className="mt-2 max-h-36 space-y-1 overflow-auto pr-1">
									{visibleYearOptions.map((year) => (
										<label
											key={year}
											className="flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-slate-800 text-sm hover:bg-[#e9f1fc]"
										>
											<input
												type="checkbox"
												checked={selectedStartYears.includes(year)}
												onChange={() => toggleStartYear(year)}
												className="h-3.5 w-3.5"
											/>
											<span>{year}</span>
										</label>
									))}
								</div>
							</div>
						</details>

						{activeRibbonTab !== "inspection-data" ? (
							<details className="group relative">
								<summary className="inline-flex w-44 cursor-pointer list-none items-center justify-between rounded-md border border-[#b6c6dc] bg-[#f8fbff] px-3 py-2 font-medium text-slate-800 text-sm">
									<span className="truncate">Statystyka: {trendMode === "average" ? "Średnia" : "Mediana"}</span>
									<ChevronDown size={14} className="shrink-0 text-slate-500 transition-transform duration-200 group-open:rotate-180" />
								</summary>
								<div className="absolute left-0 z-30 mt-2 w-44 rounded-lg border border-[#b6c6dc] bg-[#f8fbff] p-1.5 shadow-[0_14px_28px_rgba(2,8,23,0.24)]">
									<button
										type="button"
										onClick={(event) => selectTrendMode("median", event)}
										className={`flex w-full items-center rounded-md px-2.5 py-2 text-left font-medium text-sm transition-colors ${
											trendMode === "median" ? "bg-[#dce9fa] text-slate-900" : "text-slate-800 hover:bg-[#e9f1fc]"
										}`}
									>
										Mediana
									</button>
									<button
										type="button"
										onClick={(event) => selectTrendMode("average", event)}
										className={`mt-1 flex w-full items-center rounded-md px-2.5 py-2 text-left font-medium text-sm transition-colors ${
											trendMode === "average" ? "bg-[#dce9fa] text-slate-900" : "text-slate-800 hover:bg-[#e9f1fc]"
										}`}
									>
										Średnia
									</button>
								</div>
							</details>
						) : null}
					</div>

					<div className="flex items-center">
						<button
							type="button"
							onClick={clearFilters}
							disabled={!hasActiveFilters}
							className={`inline-flex h-7 items-center rounded px-1.5 transition-colors disabled:cursor-not-allowed ${
								hasActiveFilters
									? "font-semibold text-blue-300 text-sm hover:text-blue-200"
									: "font-medium text-slate-500 text-xs"
							}`}
						>
							Wyczyść filtry
						</button>
					</div>
				</div>
			</div>

			{isManagerView ? (
				<div className="flex flex-wrap items-end gap-2 border-[#2a4772] border-b">
					<button
						type="button"
						onClick={() => setActiveRibbonTab("inspection-data")}
						className={`-mb-px inline-flex h-9 items-center rounded-t-md border px-3.5 font-semibold text-sm transition-colors ${
							activeRibbonTab === "inspection-data"
								? "border-[#8fb6ee] border-b-[#101f39] bg-[#f8fbff] text-slate-900"
								: "border-transparent bg-transparent text-white hover:bg-[#18365a]/35 hover:text-white"
						}`}
					>
						Dane Inspekcji
					</button>
					<button
						type="button"
						onClick={() => setActiveRibbonTab("time-summary")}
						className={`-mb-px inline-flex h-9 items-center rounded-t-md border px-3.5 font-semibold text-sm transition-colors ${
							activeRibbonTab === "time-summary"
								? "border-[#8fb6ee] border-b-[#101f39] bg-[#f8fbff] text-slate-900"
								: "border-transparent bg-transparent text-white hover:bg-[#18365a]/35 hover:text-white"
						}`}
					>
						Zestawienie czasu
					</button>
					<button
						type="button"
						onClick={() => setActiveRibbonTab("visualization")}
						className={`-mb-px inline-flex h-9 items-center rounded-t-md border px-3.5 font-semibold text-sm transition-colors ${
							activeRibbonTab === "visualization"
								? "border-[#8fb6ee] border-b-[#101f39] bg-[#f8fbff] text-slate-900"
								: "border-transparent bg-transparent text-white hover:bg-[#18365a]/35 hover:text-white"
						}`}
					>
						Wizualizacja czasów
					</button>
				</div>
			) : (
				<div className="flex flex-wrap items-end gap-2 border-[#2a4772] border-b">
					<button
						type="button"
						onClick={() => setActiveRibbonTab("inspection-data")}
						className={`-mb-px inline-flex h-9 items-center rounded-t-md border px-3.5 font-semibold text-sm transition-colors ${
							activeRibbonTab === "inspection-data"
								? "border-[#8fb6ee] border-b-[#101f39] bg-[#f8fbff] text-slate-900"
								: "border-transparent bg-transparent text-white hover:bg-[#18365a]/35 hover:text-white"
						}`}
					>
						Moje inspekcje
					</button>
					<button
						type="button"
						onClick={() => setActiveRibbonTab("time-summary")}
						className={`-mb-px inline-flex h-9 items-center rounded-t-md border px-3.5 font-semibold text-sm transition-colors ${
							activeRibbonTab === "time-summary"
								? "border-[#8fb6ee] border-b-[#101f39] bg-[#f8fbff] text-slate-900"
								: "border-transparent bg-transparent text-white hover:bg-[#18365a]/35 hover:text-white"
						}`}
					>
						Zestawienie czasu
					</button>
				</div>
			)}

			{activeRibbonTab === "inspection-data" ? (
				<div className="flex min-h-0 flex-1 flex-col space-y-2">
					{!isManagerView ? (
						<div className="px-1">
							<h3 className="font-semibold text-[#dce9fa] text-sm">1. Dane szczegółowe</h3>
						</div>
					) : null}

				<div className="relative subtle-horizontal-scroll subtle-vertical-scroll table-scroll-gutter-right min-h-0 flex-1 w-full max-w-full overflow-x-auto overflow-y-auto rounded-xl border border-slate-300 bg-white shadow-[0_10px_28px_rgba(2,8,23,0.18)]">
					{isLoading ? <TableLoadingOverlay /> : null}
					<table className="w-full min-w-max border-collapse text-slate-900 text-sm">
					<thead>
						<tr className="bg-slate-100 text-slate-800">
							{DETAIL_COLUMNS.map((column) => (
								<th
									key={column.key}
									className={`${column.width} sticky top-0 z-20 whitespace-nowrap border-slate-300 border-b bg-slate-100 px-3 py-2 text-left font-semibold`}
								>
									{column.label}
								</th>
							))}
						</tr>
					</thead>

					<tbody>
						{sortedDetailRows.map((row, index) => (
							<tr
								key={`${row.nazwaPodmiotu}-${row.kontrola}-${index}`}
								className="border-slate-200 border-b bg-white last:border-b-0"
							>
								{DETAIL_COLUMNS.map((column) => (
									<td
										key={`${row.nazwaPodmiotu}-${column.key}-${index}`}
										className={`${column.width} px-3 py-2.5 align-top whitespace-normal wrap-break-word text-slate-900`}
									>
										{row[column.key]}
									</td>
								))}
							</tr>
						))}

						{!isLoading && sortedDetailRows.length === 0 ? (
							<tr>
								<td colSpan={DETAIL_COLUMNS.length} className="px-3 py-6 text-center text-slate-500 text-sm">
									Brak rekordów raportu.
								</td>
							</tr>
						) : null}
					</tbody>
					</table>
				</div>
				</div>
			) : null}

			{activeRibbonTab === "time-summary" ? (
				<div className="subtle-vertical-scroll flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
			<div className="space-y-2">
				<div className="px-1">
					<h3 className="font-semibold text-[#dce9fa] text-sm">
						{isManagerView ? yearlyTimeSectionTitle : "2. Zestawienie statystyczne czasu"}
					</h3>
					{!isManagerView ? (
						<p className="text-[#a7c0df] text-xs">{yearlyTimeSectionTitle}</p>
					) : null}
				</div>
				<div
					className={`subtle-horizontal-scroll subtle-vertical-scroll table-scroll-gutter-right w-full max-w-full overflow-x-auto rounded-xl border border-slate-300 bg-white shadow-[0_10px_28px_rgba(2,8,23,0.18)] ${
						shouldScrollSummaryTable ? "max-h-[30rem] overflow-y-auto" : "overflow-y-visible"
					}`}
				>
					{isLoading ? <TableLoadingOverlay /> : null}
					<table className="w-full min-w-max border-collapse text-slate-900 text-sm">
					<thead>
						<tr className="bg-slate-100 text-slate-800">
							<th className="sticky top-0 z-20 min-w-24 whitespace-nowrap border-slate-300 border-b bg-slate-100 px-3 py-2 text-left font-semibold">
								Zespół
							</th>
							{summaryPivotColumns.map((column) => (
								<th
									key={column.key}
									className="sticky top-0 z-20 min-w-24 whitespace-nowrap border-slate-300 border-b bg-slate-100 px-3 py-2 text-left font-semibold"
								>
									{column.label}
								</th>
							))}
						</tr>
					</thead>

					<tbody>
						{summaryPivotRows.map((row, index) => (
							<tr
								key={`${row.zespol}-${index}`}
								className={`border-slate-200 border-b last:border-b-0 ${
									row.zespol === "Departament" ? "bg-slate-50" : "bg-white"
								}`}
							>
							<td className="min-w-24 px-3 py-2.5 text-slate-900">{row.zespol}</td>
							{summaryPivotColumns.map((column) => (
								<td
									key={`${row.zespol}-${index}-${column.key}`}
									className="min-w-24 px-3 py-2.5 text-slate-900"
								>
									{formatSummaryCellValue(row.values[column.key] ?? null)}
								</td>
							))}
							</tr>
						))}

						{!isLoading && (summaryPivotRows.length === 0 || summaryPivotColumns.length === 0) ? (
							<tr>
								<td colSpan={Math.max(2, summaryPivotColumns.length + 1)} className="px-3 py-6 text-center text-slate-500 text-sm">
									Brak danych do agregacji.
								</td>
							</tr>
						) : null}

						{isLoading ? (
							<tr>
								<td colSpan={Math.max(2, summaryPivotColumns.length + 1)} className="px-3 py-6 text-center text-slate-500 text-sm">
									Ładowanie danych raportu...
								</td>
							</tr>
						) : null}
					</tbody>
				</table>
				</div>
			</div>

			{!isManagerView ? (
				<div className="space-y-2">
					<div className="px-1">
						<h3 className="font-semibold text-[#dce9fa] text-sm">Wykres</h3>
					</div>

					<div className="relative rounded-xl border border-[#becadd] bg-[#f8fbff] p-3 shadow-[0_10px_28px_rgba(2,8,23,0.18)]">
						{isLoading ? <TableLoadingOverlay /> : null}
						<div className="mb-3 flex flex-wrap items-center justify-between gap-2">
							<h3 className="font-semibold text-[#133259] text-sm">
								{yearlyTimeSectionTitle}
							</h3>
							<div className="flex flex-wrap items-center gap-3 text-[#33577f] text-xs">
								<label className="inline-flex cursor-pointer items-center gap-1.5">
									<input
										type="checkbox"
										checked={showHistogram}
										onChange={(event) => setShowHistogram(event.target.checked)}
										className="h-3.5 w-3.5"
									/>
									<span>Dodaj histogram</span>
								</label>
								<label className="inline-flex cursor-pointer items-center gap-1.5">
									<input
										type="checkbox"
										checked={showScatterPoints}
										onChange={(event) => setShowScatterPoints(event.target.checked)}
										className="h-3.5 w-3.5"
									/>
									<span>Dodaj punkty</span>
								</label>
							</div>
						</div>

						{scatterRowsWithDisplayNames.length > 0 ? (
							<div
								className={`h-[30rem] w-full ${
									shouldEnableChartScroll
										? "subtle-horizontal-scroll overflow-x-auto overflow-y-hidden"
										: "overflow-hidden"
								}`}
							>
								<div
									className="relative h-full"
									onClick={handleChartBackgroundClick}
									style={
										shouldEnableChartScroll
											? { minWidth: `${chartScrollableWidthPx}px` }
											: undefined
									}
								>
								<ResponsiveContainer width="100%" height="100%">
									<ComposedChart
										data={chartTrendData}
										margin={{ top: 8, right: 12, left: 8, bottom: 28 }}
									>
										<CartesianGrid stroke="#d5deea" strokeDasharray="3 3" />
										{yearRangeAreas.map((area) => (
											<ReferenceArea
												key={`shell-summary-${area.year}`}
												x1={area.x - YEAR_BOX_HALF_WIDTH}
												x2={area.x + YEAR_BOX_HALF_WIDTH}
												y1={area.y1}
												y2={area.y2}
												fill="#8dbaf5"
												fillOpacity={0.24}
												stroke="#3f6ea8"
												strokeOpacity={0.75}
											/>
										))}
										{showHistogram
											? histogramAreas.map((area) => (
												<ReferenceArea
													key={area.key}
													x1={area.x1}
													x2={area.x2}
													y1={area.y1}
													y2={area.y2}
													fill="#3f7cc5"
													fillOpacity={area.opacity}
													stroke="#2f5f95"
													strokeOpacity={0.22}
												/>
											))
											: null}
										<XAxis
											type="number"
											dataKey="x"
											allowDecimals={false}
											tick={{ fill: "#27486d", fontSize: 12 }}
											ticks={yearBuckets.map((_, index) => index)}
											tickFormatter={(value) => String(yearBuckets[value] ?? "")}
											domain={xAxisDomain}
											label={{ value: "Rok", position: "bottom", offset: 8, fill: "#27486d" }}
										/>
										<YAxis
											type="number"
											allowDecimals={false}
											tick={{ fill: "#27486d", fontSize: 12 }}
											domain={yAxisDomain}
											label={{ value: "Czas (dni)", angle: -90, position: "insideLeft", fill: "#27486d" }}
										/>
										<Line
											type="monotone"
											dataKey="trend"
											stroke="#255087"
											strokeWidth={3}
											dot={renderTrendDot}
											connectNulls={false}
											isAnimationActive={false}
										/>
										{showScatterPoints ? (
											<Scatter
												data={chartPointsData}
												dataKey="time"
												fill="#5d8fc9"
													shape={renderScatterPoint}
											/>
										) : null}
									</ComposedChart>
								</ResponsiveContainer>
								{chartClickTooltip ? (
									<div
										className="pointer-events-none absolute z-20 max-w-72 rounded-md border border-[#b6c6dc] bg-white px-3 py-2 text-[#1a3559] text-xs shadow-[0_8px_18px_rgba(2,8,23,0.18)]"
										style={{
											left: `${chartClickTooltip.x + 10}px`,
											top: `${Math.max(8, chartClickTooltip.y - 12)}px`,
										}}
									>
										{chartClickTooltip.kind === "point" ? (
											<>
												<p><strong>Nazwa podmiotu:</strong> {chartClickTooltip.data.nazwaPodmiotu}</p>
												<p><strong>Kontrola:</strong> {chartClickTooltip.data.kontrola}</p>
												<p><strong>Osoba kierująca:</strong> {chartClickTooltip.data.osobaKierujaca}</p>
												<p><strong>Zespół:</strong> {chartClickTooltip.data.zespol}</p>
												<p><strong>Czas:</strong> {chartClickTooltip.data.time}</p>
											</>
										) : (
											<>
												<p><strong>Rok:</strong> {chartClickTooltip.year}</p>
												<p>
													<strong>{trendMode === "average" ? "Średnia" : "Mediana"}:</strong>{" "}
													{chartClickTooltip.trend === null ? "-" : chartClickTooltip.trend.toFixed(1)}
												</p>
											</>
										)}
									</div>
								) : null}
								</div>
							</div>
						) : (
							<p className="py-8 text-center text-[#4b6484] text-sm">
								Brak danych liczbowych czasu do wyświetlenia wykresu.
							</p>
						)}
					</div>
				</div>
			) : null}

			<div className="space-y-2">
				<div className="px-1">
					<h3 className="font-semibold text-[#dce9fa] text-sm">{yearlyCountSectionTitle}</h3>
				</div>
				<div className="relative subtle-horizontal-scroll subtle-vertical-scroll table-scroll-gutter-right w-full max-w-full overflow-x-auto overflow-y-auto rounded-xl border border-slate-300 bg-white shadow-[0_10px_28px_rgba(2,8,23,0.18)]">
					{isLoading ? <TableLoadingOverlay /> : null}
					<table className="w-full min-w-max border-collapse text-slate-900 text-sm">
						<thead>
							<tr className="bg-slate-100 text-slate-800">
								<th className="sticky top-0 z-20 min-w-32 whitespace-nowrap border-slate-300 border-b bg-slate-100 px-3 py-2 text-left font-semibold">
									Zespół
								</th>
								{resolvedYearCountColumns.map((year) => (
									<th
										key={`count-${year}`}
										className="sticky top-0 z-20 min-w-24 whitespace-nowrap border-slate-300 border-b bg-slate-100 px-3 py-2 text-left font-semibold"
									>
										{year}
									</th>
								))}
							</tr>
						</thead>

						<tbody>
							{resolvedYearCountRows.map((row, index) => (
								<tr
									key={`year-count-${row.label}-${index}`}
									className="border-slate-200 border-b bg-white last:border-b-0"
								>
									<td className="min-w-32 px-3 py-2.5 text-slate-900">{row.label}</td>
									{resolvedYearCountColumns.map((year) => (
										<td
											key={`year-count-${row.label}-${year}`}
											className="min-w-24 px-3 py-2.5 text-slate-900"
										>
											{formatSummaryCellValue(row.values[year] ?? 0)}
										</td>
									))}
								</tr>
							))}

							{!isLoading && !hasYearCountData ? (
								<tr>
									<td colSpan={Math.max(2, resolvedYearCountColumns.length + 1)} className="px-3 py-6 text-center text-slate-500 text-sm">
										Brak danych liczności wg roku.
									</td>
								</tr>
							) : null}

							{isLoading ? (
								<tr>
									<td colSpan={Math.max(2, resolvedYearCountColumns.length + 1)} className="px-3 py-6 text-center text-slate-500 text-sm">
										Ładowanie danych raportu...
									</td>
								</tr>
							) : null}
						</tbody>
					</table>
				</div>
			</div>

			<div className="space-y-2">
				<div className="px-1">
					<h3 className="font-semibold text-[#dce9fa] text-sm">{allYearsSectionTitle}</h3>
					<p className="text-[#a7c0df] text-xs">Główna statystyka: {metricColumnLabel}</p>
				</div>
				<div className="relative subtle-horizontal-scroll subtle-vertical-scroll table-scroll-gutter-right w-full max-w-full overflow-x-auto overflow-y-auto rounded-xl border border-slate-300 bg-white shadow-[0_10px_28px_rgba(2,8,23,0.18)]">
					{isLoading ? <TableLoadingOverlay /> : null}
					<table className="w-full min-w-max border-collapse text-slate-900 text-sm">
						<thead>
							<tr className="bg-slate-100 text-slate-800">
								{resolvedOverallColumns.map((column) => (
									<th
										key={column.key}
										className="sticky top-0 z-20 min-w-24 whitespace-nowrap border-slate-300 border-b bg-slate-100 px-3 py-2 text-left font-semibold"
									>
										{column.label}
									</th>
								))}
							</tr>
						</thead>

						<tbody>
							{resolvedOverallRows.map((row, index) => {
								const teamValue = String(row.zespol ?? "");
								const rowClassName =
									teamValue === "Departament"
										? "border-slate-200 border-b bg-slate-50 last:border-b-0"
										: "border-slate-200 border-b bg-white last:border-b-0";

								return (
									<tr key={`overall-${index}`} className={rowClassName}>
										{resolvedOverallColumns.map((column) => (
											<td key={`overall-${index}-${column.key}`} className="min-w-24 px-3 py-2.5 text-slate-900">
												{formatSummaryCellValue(row[column.key] ?? null)}
											</td>
										))}
									</tr>
								);
							})}

							{!isLoading && (resolvedOverallRows.length === 0 || resolvedOverallColumns.length === 0) ? (
								<tr>
									<td colSpan={Math.max(1, resolvedOverallColumns.length)} className="px-3 py-6 text-center text-slate-500 text-sm">
										Brak danych do zestawienia ogólnego.
									</td>
								</tr>
							) : null}

							{isLoading ? (
								<tr>
									<td colSpan={Math.max(1, resolvedOverallColumns.length)} className="px-3 py-6 text-center text-slate-500 text-sm">
										Ładowanie danych raportu...
									</td>
								</tr>
							) : null}
						</tbody>
					</table>
				</div>
			</div>
				</div>
			) : null}

			{isManagerView && activeRibbonTab === "visualization" ? (
				<div className="space-y-2">
					<div className="px-1">
						<h3 className="font-semibold text-[#dce9fa] text-sm">Wykres porównawczy</h3>
					</div>

					<div className="relative rounded-xl border border-[#becadd] bg-[#f8fbff] p-3 shadow-[0_10px_28px_rgba(2,8,23,0.18)]">
						{isLoading ? <TableLoadingOverlay /> : null}
					<div className="mb-3 flex flex-wrap items-center justify-between gap-2">
						<h3 className="font-semibold text-[#133259] text-sm">
							{yearlyTimeSectionTitle}
						</h3>
						<div className="flex flex-wrap items-center gap-3 text-[#33577f] text-xs">
							<label className="inline-flex cursor-pointer items-center gap-1.5">
								<input
									type="checkbox"
									checked={showHistogram}
									onChange={(event) => setShowHistogram(event.target.checked)}
									className="h-3.5 w-3.5"
								/>
								<span>Dodaj histogram</span>
							</label>
							<label className="inline-flex cursor-pointer items-center gap-1.5">
								<input
									type="checkbox"
									checked={showScatterPoints}
									onChange={(event) => setShowScatterPoints(event.target.checked)}
									className="h-3.5 w-3.5"
								/>
								<span>Dodaj punkty</span>
							</label>
						</div>
					</div>

					{scatterRowsWithDisplayNames.length > 0 ? (
						<div
							className={`h-[30rem] w-full ${
								shouldEnableChartScroll
									? "subtle-horizontal-scroll overflow-x-auto overflow-y-hidden"
									: "overflow-hidden"
							}`}
						>
							<div
								className="relative h-full"
								onClick={handleChartBackgroundClick}
								style={
									shouldEnableChartScroll
										? { minWidth: `${chartScrollableWidthPx}px` }
										: undefined
								}
							>
							<ResponsiveContainer width="100%" height="100%">
								<ComposedChart
									data={chartTrendData}
									margin={{ top: 8, right: 12, left: 8, bottom: 28 }}
								>
									<CartesianGrid stroke="#d5deea" strokeDasharray="3 3" />
									{yearRangeAreas.map((area) => (
										<ReferenceArea
											key={`shell-visualization-${area.year}`}
											x1={area.x - YEAR_BOX_HALF_WIDTH}
											x2={area.x + YEAR_BOX_HALF_WIDTH}
											y1={area.y1}
											y2={area.y2}
											fill="#8dbaf5"
											fillOpacity={0.24}
											stroke="#3f6ea8"
											strokeOpacity={0.75}
										/>
									))}
									{showHistogram
										? histogramAreas.map((area) => (
											<ReferenceArea
												key={area.key}
												x1={area.x1}
												x2={area.x2}
												y1={area.y1}
												y2={area.y2}
												fill="#3f7cc5"
												fillOpacity={area.opacity}
												stroke="#2f5f95"
												strokeOpacity={0.22}
											/>
										))
										: null}
									<XAxis
										type="number"
										dataKey="x"
										allowDecimals={false}
										tick={{ fill: "#27486d", fontSize: 12 }}
										ticks={yearBuckets.map((_, index) => index)}
										tickFormatter={(value) => String(yearBuckets[value] ?? "")}
										domain={xAxisDomain}
										label={{ value: "Rok", position: "bottom", offset: 8, fill: "#27486d" }}
									/>
									<YAxis
										type="number"
										allowDecimals={false}
										tick={{ fill: "#27486d", fontSize: 12 }}
										domain={yAxisDomain}
										label={{ value: "Czas (dni)", angle: -90, position: "insideLeft", fill: "#27486d" }}
									/>
									<Line
										type="monotone"
										dataKey="trend"
										stroke="#255087"
										strokeWidth={3}
										dot={renderTrendDot}
										connectNulls={false}
										isAnimationActive={false}
									/>
									{showScatterPoints ? (
										<Scatter
											data={chartPointsData}
											dataKey="time"
											fill="#5d8fc9"
												shape={renderScatterPoint}
										/>
									) : null}
								</ComposedChart>
							</ResponsiveContainer>
							{chartClickTooltip ? (
								<div
									className="pointer-events-none absolute z-20 max-w-72 rounded-md border border-[#b6c6dc] bg-white px-3 py-2 text-[#1a3559] text-xs shadow-[0_8px_18px_rgba(2,8,23,0.18)]"
									style={{
										left: `${chartClickTooltip.x + 10}px`,
										top: `${Math.max(8, chartClickTooltip.y - 12)}px`,
									}}
								>
									{chartClickTooltip.kind === "point" ? (
										<>
											<p><strong>Nazwa podmiotu:</strong> {chartClickTooltip.data.nazwaPodmiotu}</p>
											<p><strong>Kontrola:</strong> {chartClickTooltip.data.kontrola}</p>
											<p><strong>Osoba kierująca:</strong> {chartClickTooltip.data.osobaKierujaca}</p>
											<p><strong>Zespół:</strong> {chartClickTooltip.data.zespol}</p>
											<p><strong>Czas:</strong> {chartClickTooltip.data.time}</p>
										</>
									) : (
										<>
											<p><strong>Rok:</strong> {chartClickTooltip.year}</p>
											<p>
												<strong>{trendMode === "average" ? "Średnia" : "Mediana"}:</strong>{" "}
												{chartClickTooltip.trend === null ? "-" : chartClickTooltip.trend.toFixed(1)}
											</p>
										</>
									)}
								</div>
							) : null}
							</div>
						</div>
					) : (
						<p className="py-8 text-center text-[#4b6484] text-sm">
							Brak danych liczbowych czasu do wyświetlenia wykresu.
						</p>
					)}
					</div>
				</div>
			) : null}

		</section>
	);
}

type SingleReportPanelProps = {
	operatorLogin: string;
};

export function ProtocolTimePanel({ operatorLogin }: SingleReportPanelProps) {
	return (
		<TimeReportPanel
			operatorLogin={operatorLogin}
			title="Czas protokołu"
			inspectionType="K"
		/>
	);
}

export function StatementTimePanel({ operatorLogin }: SingleReportPanelProps) {
	return (
		<TimeReportPanel
			operatorLogin={operatorLogin}
			title="Czas sprawozdania"
			inspectionType="W"
		/>
	);
}
