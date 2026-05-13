import { ChevronDown, Info } from "lucide-react";
import { useRef } from "react";

import type { TableSortDirection } from "@/shared/types/table";

type TableHeaderColumn<TColumnKey extends string> = {
	key: TColumnKey;
	label: string;
	tooltip?: string;
};

type TableHeaderWithFiltersProps<TColumnKey extends string> = {
	visibleColumns: TableHeaderColumn<TColumnKey>[];
	sortColumnKey: TColumnKey | null;
	sortDirection: TableSortDirection | null;
	advancedFilters: Partial<Record<TColumnKey, string[]>>;
	columnFilters: Partial<Record<TColumnKey, string>>;
	onSortByColumn: (columnKey: TColumnKey) => void;
	onOpenAdvancedFilter: (columnKey: TColumnKey, triggerElement: HTMLElement) => void;
	onFilterChange: (columnKey: TColumnKey, value: string) => void;
	columnWidths?: Partial<Record<TColumnKey, number>>;
	columnMinWidths?: Partial<Record<TColumnKey, number>>;
	onResizeColumn?: (columnKey: TColumnKey, width: number) => void;
	minColumnWidth?: number;
	compact?: boolean;
};

export function TableHeaderWithFilters<TColumnKey extends string>({
	visibleColumns,
	sortColumnKey,
	sortDirection,
	advancedFilters,
	columnFilters,
	onSortByColumn,
	onOpenAdvancedFilter,
	onFilterChange,
	columnWidths,
	columnMinWidths,
	onResizeColumn,
	minColumnWidth = 110,
	compact = false,
}: TableHeaderWithFiltersProps<TColumnKey>) {
	const resizeCleanupRef = useRef<(() => void) | null>(null);

	const startColumnResize = (
		event: React.MouseEvent<HTMLDivElement>,
		columnKey: TColumnKey,
		startWidth: number,
	) => {
		if (!onResizeColumn) {
			return;
		}

		const effectiveMinWidth = Math.max(
			minColumnWidth,
			columnMinWidths?.[columnKey] ?? minColumnWidth,
		);

		event.preventDefault();
		event.stopPropagation();

		const initialX = event.clientX;
		const baseWidth = Number.isFinite(startWidth) && startWidth > 0
			? startWidth
			: effectiveMinWidth;

		const handleMouseMove = (moveEvent: MouseEvent) => {
			const deltaX = moveEvent.clientX - initialX;
			const nextWidth = Math.max(effectiveMinWidth, Math.round(baseWidth + deltaX));
			onResizeColumn(columnKey, nextWidth);
		};

		const cleanup = () => {
			document.removeEventListener("mousemove", handleMouseMove);
			document.removeEventListener("mouseup", cleanup);
			resizeCleanupRef.current = null;
		};

		resizeCleanupRef.current?.();
		resizeCleanupRef.current = cleanup;
		document.addEventListener("mousemove", handleMouseMove);
		document.addEventListener("mouseup", cleanup);
	};

	return (
		<thead>
			<tr className="bg-slate-100 text-slate-800">
				{visibleColumns.map((column) => {
					const columnWidth = columnWidths?.[column.key];

					return (
					<th
						key={column.key}
						className={`sticky top-0 z-10 whitespace-nowrap border-slate-300 border-b bg-slate-100 text-left font-semibold ${
							compact ? "px-2 py-1.5 text-xs" : "px-3 py-2"
						}`}
						style={
							columnWidth
								? { width: columnWidth, minWidth: columnWidth, maxWidth: columnWidth }
								: undefined
						}
					>
						<div className="group relative flex items-center gap-1.5 pr-8">
							<button
								type="button"
								onClick={() => onSortByColumn(column.key)}
								className="inline-flex min-w-0 items-center gap-1.5 rounded-sm px-0.5 py-0.5 text-left hover:text-[#1f4e8c]"
							>
								<span className="truncate">{column.label}</span>
								{column.tooltip ? (
									<Info
										size={compact ? 12 : 13}
										className="text-slate-500"
										aria-hidden="true"
									/>
								) : null}
								<span className="text-[10px] text-slate-500 leading-none">
									{sortColumnKey === column.key
										? sortDirection === "asc"
											? "▲"
											: sortDirection === "desc"
												? "▼"
												: "↕"
										: "↕"}
								</span>
							</button>

							{column.tooltip ? (
								<div className="pointer-events-none absolute top-full left-0 z-30 mt-1 hidden w-104 max-w-[90vw] whitespace-pre-line rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-700 text-sm leading-5 shadow-lg group-hover:block">
									{column.tooltip}
								</div>
							) : null}

							<button
								type="button"
								onClick={(event) => {
									event.stopPropagation();
									onOpenAdvancedFilter(column.key, event.currentTarget);
								}}
								aria-label={
									column.tooltip
										? `Filtruj kolumnę ${column.label}. ${column.tooltip}`
										: `Filtruj kolumnę ${column.label}`
								}
								className={`absolute top-1/2 right-0 inline-flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded border transition-colors ${
									(advancedFilters[column.key] ?? []).length > 0
										? "border-blue-400 bg-blue-50 text-blue-700"
										: "border-slate-300 bg-white text-slate-600 hover:bg-slate-100"
								}`}
							>
								<ChevronDown size={compact ? 11 : 12} />
							</button>

							{onResizeColumn ? (
								<div
									role="separator"
									aria-orientation="vertical"
									onMouseDown={(event) =>
										startColumnResize(
											event,
											column.key,
											columnWidth ?? event.currentTarget.parentElement?.getBoundingClientRect().width ?? minColumnWidth,
										)
									}
									className="absolute top-0 -right-2 z-20 h-full w-3 cursor-col-resize rounded-sm before:absolute before:top-1/2 before:left-1/2 before:h-7 before:w-px before:-translate-x-1/2 before:-translate-y-1/2 before:bg-slate-400 before:content-[''] hover:bg-blue-100/60 hover:before:w-0.5 hover:before:bg-blue-600"
									title="Przeciągnij, aby zmienić szerokość kolumny"
								/>
							) : null}
						</div>
					</th>
					);
				})}
			</tr>

			<tr className="bg-white text-slate-700">
				{visibleColumns.map((column) => {
					const columnWidth = columnWidths?.[column.key];

					return (
					<th
						key={`${column.key}-filter`}
						className={`overflow-hidden border-slate-200 border-b bg-white ${
							compact ? "px-1.5 py-1.5" : "px-2 py-2"
						}`}
						style={
							columnWidth
								? { width: columnWidth, minWidth: columnWidth, maxWidth: columnWidth }
								: undefined
						}
					>
						<input
							type="text"
							value={columnFilters[column.key] ?? ""}
							onChange={(event) =>
								onFilterChange(column.key, event.target.value)
							}
							placeholder="Filtruj..."
							className={`w-full rounded-md border border-slate-300 bg-white text-slate-800 outline-none transition-colors focus:border-blue-400 ${
								compact ? "px-1.5 py-0.5 text-[11px]" : "px-2 py-1 text-xs"
							}`}
						/>
					</th>
					);
				})}
			</tr>
		</thead>
	);
}
