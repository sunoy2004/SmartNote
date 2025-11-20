import { FileText } from "lucide-react";

interface NotesHeaderProps {
  notesCount: number;
}

export const NotesHeader = ({ notesCount }: NotesHeaderProps) => {
  return (
    <header className="border-b border-border bg-card">
      <div className="container mx-auto px-4 py-6 max-w-4xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-8 h-8 text-primary" />
            <div>
              <h1 className="text-2xl font-bold text-foreground">AI Notes</h1>
              <p className="text-sm text-muted-foreground">Voice transcription & summarization</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-primary">{notesCount}</div>
            <div className="text-xs text-muted-foreground">Notes</div>
          </div>
        </div>
      </div>
    </header>
  );
};
