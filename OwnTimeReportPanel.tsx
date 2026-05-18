"use client";

import {
	CartesianGrid,
	Legend,
	Line,
	LineChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";
import { ChevronDown } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { fetchInspectionsTimeAnalytics } from "@/features/reports/api";
import type {
	ReportInspectionDetailedRow,
	TimeReportOverallColumn,
	TimeReportOverallRow,
	TimeReportSummaryPivotRow,
	TimeReportTrendMode,
} from "@/features/reports/types";

const INSPECTIONS_CHANGED_EVENT = "inspections:changed";

type OwnTimeReportPanelProps = {
	operatorLogin: string;
};

type OwnTimeTab = "inspection-data" | "time-summary";

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

export function OwnTimeReportPanel({ operatorLogin }: OwnTimeReportPanelProps) {
	const [inspectionType, setInspectionType] = useState<"K" | "W">("K");
	const [trendMode, setTrendMode] = useState<TimeReportTrendMode>("average");
	const [detailRows, setDetailRows] = useState<ReportInspectionDetailedRow[]>([]);
	const [summaryPivotYears, setSummaryPivotYears] = useState<string[]>([]);
	const [summaryPivotRows, setSummaryPivotRows] = useState<TimeReportSummaryPivotRow[]>([]);
	const [selectedMetricLabel, setSelectedMetricLabel] = useState("Średni czas");
	const [overallColumns, setOverallColumns] = useState<TimeReportOverallColumn[]>([]);
	const [overallRows, setOverallRows] = useState<TimeReportOverallRow[]>([]);
	const [baseCount, setBaseCount] = useState(0);
	const [filteredCount, setFilteredCount] = useState(0);
	const [activeTab, setActiveTab] = useState<OwnTimeTab>("inspection-data");
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const filtersRef = useRef<HTMLDivElement | null>(null);

	const loadReport = useCallback(async () => {
		setIsLoading(true);
		setError(null);

		const result = await fetchInspectionsTimeAnalytics(operatorLogin, {
			inspectionType,
			trendMode,
			teams: [],
			years: [],
		});

		if (!result.ok) {
			setDetailRows([]);
			setSummaryPivotYears([]);
			setSummaryPivotRows([]);
			setSelectedMetricLabel(trendMode === "average" ? "Średni czas" : "Mediana czasu");
			setOverallColumns([]);
			setOverallRows([]);
			setBaseCount(0);
			setFilteredCount(0);
			setError(result.error);
			setIsLoading(false);
			return;
		}

		setDetailRows(result.data.detailRows);
		setSummaryPivotYears(result.data.summaryPivotYears);
		setSummaryPivotRows(result.data.summaryPivotRows);
		setSelectedMetricLabel(result.data.selectedMetricLabel);
		setOverallColumns(result.data.overallColumns);
		setOverallRows(result.data.overallRows);
		setBaseCount(result.data.baseCount);
		setFilteredCount(result.data.filteredCount);
		setIsLoading(false);
	}, [inspectionType, operatorLogin, trendMode]);

	useEffect(() => {
		void loadReport();
	}, [loadReport]);

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

	const summaryYearsWithoutAll = useMemo(
		() => summaryPivotYears.filter((year) => year !== "allYears"),
		[summaryPivotYears],
	);

	const formatMetric = (value: number) =>
		Number.isFinite(value) ? value.toFixed(1) : "0.0";

	const formatSummaryCellValue = (value: string | number | null) => {
		if (value === null || value === undefined || value === "") {
			return "-";
		}

		if (typeof value === "number") {
			return formatMetric(value);
		}

		if (value.trim() !== "") {
			const parsed = Number(value.replace(",", "."));
			if (Number.isFinite(parsed)) {
				return formatMetric(parsed);
			}
		}

		return value;
	};

	const formatOverallCellValue = (value: string | number | null, key: string) => {
		if (value === null || value === undefined || value === "") {
			return "-";
		}

		if (typeof value === "number") {
			if (key === "average" || key === "median" || key === "metric") {
				return formatMetric(value);
			}

			return String(value);
		}

		if (
			(key === "average" || key === "median" || key === "metric") &&
			value.trim() !== ""
		) {
			const parsed = Number(value.replace(",", "."));
			if (Number.isFinite(parsed)) {
				return formatMetric(parsed);
			}
		}

		return value;
	};

	const displayedOverallColumns = useMemo(() => {
		const hasMetricColumn = overallColumns.some((column) => column.key === "metric");
		if (!hasMetricColumn) {
			return overallColumns;
		}

		return overallColumns.filter(
			(column) => column.key !== "average" && column.key !== "median",
		);
	}, [overallColumns]);

	const normalizeLabel = (value: string) => value.trim().toLowerCase();

	const toNumericMetric = (value: string | number | null) => {
		if (typeof value === "number" && Number.isFinite(value)) {
			return value;
		}

		if (typeof value === "string") {
			const parsed = Number(value.replace(",", ".").trim());
			if (Number.isFinite(parsed)) {
				return parsed;
			}
		}

		return null;
	};

	const departmentRow = useMemo(
		() =>
			summaryPivotRows.find((row) => normalizeLabel(row.zespol).includes("departament")) ??
			null,
		[summaryPivotRows],
	);

	const myTimeRow = useMemo(
		() =>
			summaryPivotRows.find((row) => {
				const label = normalizeLabel(row.zespol);
				return label.includes("mój czas") || label.includes("moj czas");
			}) ?? null,
		[summaryPivotRows],
	);

	const teamRow = useMemo(() => {
		const departmentLabel = departmentRow ? normalizeLabel(departmentRow.zespol) : "";
		const myTimeLabel = myTimeRow ? normalizeLabel(myTimeRow.zespol) : "";

		return (
			summaryPivotRows.find((row) => {
				const label = normalizeLabel(row.zespol);
				if (label === departmentLabel || label === myTimeLabel) {
					return false;
				}

				return true;
			}) ?? null
		);
	}, [departmentRow, myTimeRow, summaryPivotRows]);

	const trendChartData = useMemo(
		() =>
			summaryYearsWithoutAll.map((year) => ({
				year,
				departament: toNumericMetric(departmentRow?.values[year] ?? null),
				team: toNumericMetric(teamRow?.values[year] ?? null),
				myTime: toNumericMetric(myTimeRow?.values[year] ?? null),
			})),
		[departmentRow, myTimeRow, summaryYearsWithoutAll, teamRow],
	);

	const hasTrendChartData = useMemo(
		() => trendChartData.some((point) => point.departament !== null || point.team !== null || point.myTime !== null),
		[trendChartData],
	);

	return (
		<section className="flex h-[calc(100vh-4.4rem)] min-h-0 flex-col space-y-4 rounded-2xl border border-slate-700/70 bg-[#101f39] p-5 shadow-[0_18px_44px_rgba(2,8,23,0.34)]">
			<div className="flex flex-wrap items-center justify-between gap-2">
				<h2 className="font-semibold text-slate-100 text-xl">Mój raport czasu</h2>
			</div>

			{error ? (
				<p className="rounded-lg border border-rose-300/60 bg-rose-100/90 px-3 py-2 text-rose-800 text-sm">
					{error}
				</p>
			) : null}

			<div ref={filtersRef} className="rounded-lg border border-slate-700/70 bg-[#122c4e] p-3">
				<div className="flex flex-wrap items-center justify-between gap-2">
					<div className="flex flex-wrap items-center gap-2">
						<select
							value={inspectionType}
							onChange={(event) =>
								setInspectionType(event.target.value === "W" ? "W" : "K")
							}
							className="h-10 rounded-md border border-[#b6c6dc] bg-[#f8fbff] px-3 text-slate-800 text-sm"
						>
							<option value="K">Protokół (K)</option>
							<option value="W">Sprawozdanie (W)</option>
						</select>

						<select
							value={trendMode}
							onChange={(event) =>
								setTrendMode(
									event.target.value === "average" ? "average" : "median",
								)
							}
							className="h-10 rounded-md border border-[#b6c6dc] bg-[#f8fbff] px-3 text-slate-800 text-sm"
						>
							<option value="median">Statystyka: Mediana</option>
							<option value="average">Statystyka: Średnia</option>
						</select>

					</div>

					<div className="flex items-center gap-2">
						<span className="text-slate-300 text-sm">Pokazane rekordy: {filteredCount}/{baseCount}</span>
					</div>
				</div>
			</div>

			<div className="flex flex-wrap items-end gap-0 border-[#2a4772] border-b">
				<button
					type="button"
					onClick={() => setActiveTab("inspection-data")}
					className={`-mb-px inline-flex h-9 min-w-44 items-center justify-center rounded-t-md border px-3.5 font-semibold text-sm transition-colors ${
						activeTab === "inspection-data"
							? "border-[#8fb6ee] border-b-[#10244a] bg-[#f8fbff] text-slate-900"
							: "border-transparent bg-transparent text-white hover:bg-[#18365a]/35 hover:text-white"
					}`}
				>
					Moje inspekcje
				</button>
				<button
					type="button"
					onClick={() => setActiveTab("time-summary")}
					className={`-mb-px inline-flex h-9 min-w-44 items-center justify-center rounded-t-md border px-3.5 font-semibold text-sm transition-colors ${
						activeTab === "time-summary"
							? "border-[#8fb6ee] border-b-[#10244a] bg-[#f8fbff] text-slate-900"
							: "border-transparent bg-transparent text-white hover:bg-[#18365a]/35 hover:text-white"
					}`}
				>
					Zestawienie czasu
				</button>
			</div>

			{activeTab === "inspection-data" ? (
				<div className="flex min-h-0 flex-1 flex-col space-y-2">
					<div className="px-1">
						<h3 className="font-semibold text-[#dce9fa] text-sm">1. Dane szczegółowe</h3>
					</div>

					<div className="relative subtle-horizontal-scroll subtle-vertical-scroll min-h-0 flex-1 w-full max-w-full overflow-x-auto overflow-y-auto rounded-xl border border-slate-300 bg-white shadow-[0_10px_28px_rgba(2,8,23,0.18)]">
						{isLoading ? <TableLoadingOverlay /> : null}
						<table className="w-full min-w-max border-collapse text-slate-900 text-sm">
							<thead>
								<tr className="bg-slate-100 text-slate-800">
									<th className="min-w-56 border-slate-300 border-b px-3 py-2 text-left font-semibold">Nazwa podmiotu</th>
									<th className="min-w-40 border-slate-300 border-b px-3 py-2 text-left font-semibold">Kontrola</th>
									<th className="min-w-28 border-slate-300 border-b px-3 py-2 text-left font-semibold">Rok początku</th>
									<th className="min-w-24 border-slate-300 border-b px-3 py-2 text-left">Czas</th>
								</tr>
							</thead>
							<tbody>
								{detailRows.map((row, index) => (
									<tr key={`${row.nazwaPodmiotu}-${row.kontrola}-${index}`} className="border-slate-200 border-b bg-white last:border-b-0">
										<td className="px-3 py-2.5 text-slate-900">{row.nazwaPodmiotu}</td>
										<td className="px-3 py-2.5 text-slate-900">{row.kontrola}</td>
										<td className="px-3 py-2.5 text-slate-900">{row.rokPoczatku}</td>
										<td className="px-3 py-2.5 text-left text-slate-900">{row.czas}</td>
									</tr>
								))}
								{!isLoading && detailRows.length === 0 ? (
									<tr>
										<td colSpan={4} className="px-3 py-6 text-center text-slate-500 text-sm">Brak rekordów raportu.</td>
									</tr>
								) : null}
							</tbody>
						</table>
					</div>
				</div>
			) : null}

			{activeTab === "time-summary" ? (
				<div className="subtle-vertical-scroll min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
					<div className="px-1">
						<h3 className="font-semibold text-[#dce9fa] text-sm">2. Zestawienie statystyczne czasu</h3>
						<p className="text-[#a7c0df] text-xs">Wybrana statystyka: {selectedMetricLabel}</p>
					</div>
					<div className="relative subtle-horizontal-scroll subtle-vertical-scroll max-h-[24rem] w-full max-w-full overflow-x-auto overflow-y-auto rounded-xl border border-slate-300 bg-white shadow-[0_10px_28px_rgba(2,8,23,0.18)]">
						{isLoading ? <TableLoadingOverlay /> : null}
						<table className="w-full min-w-max border-collapse text-slate-900 text-sm">
							<thead>
								<tr className="bg-slate-100 text-slate-800">
									<th className="min-w-24 border-slate-300 border-b px-3 py-2 text-left font-semibold">Zespół</th>
									{summaryYearsWithoutAll.map((year) => (
										<th
											key={year}
											className="min-w-24 border-slate-300 border-b px-3 py-2 text-left font-semibold"
										>
											{year}
										</th>
									))}
								</tr>
							</thead>
							<tbody>
								{summaryPivotRows.map((row, index) => (
									<tr key={`${row.zespol}-${index}`} className="border-slate-200 border-b bg-white last:border-b-0">
										<td className="px-3 py-2.5 text-slate-900">{row.zespol}</td>
										{summaryYearsWithoutAll.map((year) => (
											<td
												key={`${row.zespol}-${index}-${year}`}
												className="px-3 py-2.5 text-slate-900"
											>
												{formatSummaryCellValue(row.values[year] ?? null)}
											</td>
										))}
									</tr>
								))}
								{!isLoading && (summaryPivotRows.length === 0 || summaryYearsWithoutAll.length === 0) ? (
									<tr>
										<td colSpan={Math.max(2, summaryYearsWithoutAll.length + 1)} className="px-3 py-6 text-center text-slate-500 text-sm">Brak danych do agregacji.</td>
									</tr>
								) : null}
							</tbody>
						</table>
					</div>

					<div className="space-y-2">
						<div className="px-1">
							<h3 className="font-semibold text-[#dce9fa] text-sm">Wykres</h3>
						</div>
						<div className="rounded-xl border border-[#becadd] bg-[#f8fbff] p-3 shadow-[0_10px_28px_rgba(2,8,23,0.18)]">
							<p className="mb-2 font-semibold text-[#133259] text-sm">Trend roczny ({selectedMetricLabel})</p>
							{hasTrendChartData && trendChartData.length > 0 ? (
								<div className="h-64 w-full">
									<ResponsiveContainer width="100%" height="100%">
										<LineChart data={trendChartData} margin={{ top: 8, right: 12, left: 8, bottom: 8 }}>
											<CartesianGrid stroke="#d5deea" strokeDasharray="3 3" />
											<XAxis dataKey="year" tick={{ fill: "#27486d", fontSize: 12 }} />
											<YAxis tick={{ fill: "#27486d", fontSize: 12 }} />
											<Tooltip
												formatter={(value) => {
													const rawValue = Array.isArray(value) ? value[0] : value;
													if (rawValue === null || rawValue === undefined) {
														return "-";
													}

													if (typeof rawValue === "number") {
														return formatMetric(rawValue);
													}

													const parsed = Number(String(rawValue).replace(",", ".").trim());
													return Number.isFinite(parsed) ? formatMetric(parsed) : "-";
												}}
											/>
											<Legend />
											<Line
												type="monotone"
												dataKey="departament"
												name={departmentRow?.zespol ?? "Departament"}
												stroke="#1d4ed8"
												strokeWidth={2.5}
												dot={{ r: 3 }}
												connectNulls
											/>
											<Line
												type="monotone"
												dataKey="team"
												name={teamRow?.zespol ?? "Zespół"}
												stroke="#0f766e"
												strokeWidth={2.5}
												dot={{ r: 3 }}
												connectNulls
											/>
											<Line
												type="monotone"
												dataKey="myTime"
												name={myTimeRow?.zespol ?? "Mój czas"}
												stroke="#b45309"
												strokeWidth={2.5}
												dot={{ r: 3 }}
												connectNulls
											/>
										</LineChart>
									</ResponsiveContainer>
								</div>
							) : (
								<p className="px-2 py-8 text-center text-slate-500 text-sm">Brak danych do wykresu trendu.</p>
							)}
						</div>
					</div>

					<div className="space-y-2">
						<div className="px-1">
							<h3 className="font-semibold text-[#dce9fa] text-sm">Zestawienie ogólne</h3>
							<p className="text-[#a7c0df] text-xs">Główna statystyka: {selectedMetricLabel}</p>
						</div>
						<div className="relative subtle-horizontal-scroll subtle-vertical-scroll max-h-[24rem] w-full max-w-full overflow-x-auto overflow-y-auto rounded-xl border border-slate-300 bg-white shadow-[0_10px_28px_rgba(2,8,23,0.18)]">
							{isLoading ? <TableLoadingOverlay /> : null}
							<table className="w-full min-w-max border-collapse text-slate-900 text-sm">
								<thead>
									<tr className="bg-slate-100 text-slate-800">
										{displayedOverallColumns.map((column) => (
											<th
												key={column.key}
												className="min-w-24 border-slate-300 border-b px-3 py-2 text-left font-semibold"
											>
												{column.key === "metric" ? selectedMetricLabel : column.label}
											</th>
										))}
									</tr>
								</thead>
								<tbody>
									{overallRows.map((row, rowIndex) => (
										<tr key={`overall-${rowIndex}`} className="border-slate-200 border-b bg-white last:border-b-0">
											{displayedOverallColumns.map((column) => (
												<td
													key={`overall-${rowIndex}-${column.key}`}
													className="px-3 py-2.5 text-slate-900"
												>
													{formatOverallCellValue(
														row[column.key] as string | number | null,
														column.key,
													)}
												</td>
											))}
										</tr>
									))}
									{!isLoading && (overallRows.length === 0 || displayedOverallColumns.length === 0) ? (
										<tr>
											<td colSpan={Math.max(1, displayedOverallColumns.length)} className="px-3 py-6 text-center text-slate-500 text-sm">Brak danych ogólnych.</td>
										</tr>
									) : null}
								</tbody>
							</table>
						</div>
					</div>
				</div>
			) : null}
		</section>
	);
}
