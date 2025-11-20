import { Trash2, Clock, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Note } from "@/hooks/useNotes";

interface NotesListProps {
  notes: Note[];
  onDelete: (id: string) => void;
}

export const NotesList = ({ notes, onDelete }: NotesListProps) => {
  if (notes.length === 0) {
    return (
      <div className="text-center py-12">
        <FileText className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
        <p className="text-muted-foreground">No notes yet. Start recording to create your first note!</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-foreground mb-4">Your Notes</h2>
      {notes.map((note) => (
        <Card key={note.id} className="p-6">
          <div className="flex justify-between items-start mb-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Clock className="w-4 h-4" />
              {new Date(note.timestamp).toLocaleString()}
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onDelete(note.id)}
              className="text-destructive hover:text-destructive"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>

          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-2">Summary</h3>
              <p className="text-foreground bg-muted p-3 rounded-lg">{note.summary}</p>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-foreground mb-2">Full Transcript</h3>
              <p className="text-sm text-muted-foreground p-3 bg-muted/50 rounded-lg">
                {note.transcript}
              </p>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
};
