import { Component } from '@angular/core';

import { DiffViewerComponent } from './features/diff-viewer/diff-viewer.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [DiffViewerComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {}
