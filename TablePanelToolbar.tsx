import { Download, SlidersHorizontal } from "lucide-react";
import type { ReactNode } from "react";

type TablePanelToolbarProps = {
	title: string;
	canClearFilters: boolean;
	isExporting: boolean;
	hasRowsToExport: boolean;
	onOpenViewModal: () => void;
	onClearFilters: () => void;
	onExport: () => void;
	actions?: ReactNode;
	footerActions?: ReactNode;
	exportLabel?: string;
	exportingLabel?: string;
};

export function TablePanelToolbar({
	title,
	canClearFilters,
	isExporting,
	hasRowsToExport,
	onOpenViewModal,
	onClearFilters,
	onExport,
	actions,
	footerActions,
	exportLabel = "Eksport",
	exportingLabel = "Eksportowanie...",
}: TablePanelToolbarProps) {
	return (
		<div className="mb-3 border-slate-700/70 border-b pb-2">
			<div className="flex flex-wrap items-center justify-between gap-3">
				<div className="flex flex-wrap items-center gap-3">
					<h2 className="font-semibold text-lg text-slate-100">{title}</h2>

					<button
						type="button"
						onClick={onOpenViewModal}
						className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-[#5d7eaf] bg-[#243b61] px-3 font-semibold text-slate-100 text-sm transition-colors hover:bg-[#2c4875]"
					>
						<SlidersHorizontal size={14} />
						Widok
					</button>
				</div>

				<div className="flex flex-wrap items-center gap-2">
					<button
						type="button"
						onClick={onExport}
						disabled={isExporting || !hasRowsToExport}
						className="inline-flex h-10 items-center gap-2 rounded-lg border border-[#6ea3f0] bg-[#2d4d7f] px-3.5 font-semibold text-slate-100 text-sm transition-colors hover:bg-[#375f99] disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-[#1a2946] disabled:text-slate-500"
					>
						<Download size={15} />
						{isExporting ? exportingLabel : exportLabel}
					</button>

					{actions}
				</div>
			</div>

			<div className="mt-1 flex items-center justify-end gap-3">
				<button
					type="button"
					onClick={onClearFilters}
					disabled={!canClearFilters}
					className={`inline-flex h-7 items-center rounded px-1.5 transition-colors disabled:cursor-not-allowed ${
						canClearFilters
							? "font-semibold text-blue-300 text-sm hover:text-blue-200"
							: "font-medium text-slate-500 text-xs"
					}`}
				>
					Wyczyść filtry
				</button>
				{footerActions}
			</div>
		</div>
	);
}
