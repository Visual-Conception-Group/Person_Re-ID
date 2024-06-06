import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { LoginComponent } from './Login/login.component';
import { PageNotFoundComponent } from './pagenotfound/pagenotfound.component';
import { VideoUploadComponent } from './videoupload/videoupload.component';
import {PridComponent} from "./prid/prid.component"
import { PridDemoComponent } from './prid_demo/prid_demo.component';
const routes: Routes = [
  {path : "login" , component : LoginComponent},
  { path: '',   redirectTo: '/login', pathMatch: 'full' },
  {path: "videoprocess",component:VideoUploadComponent},
  {path : "person-reid" , component : PridComponent},
  {path : "person-reid-demo" , component : PridDemoComponent},
  { path: '**', component: PageNotFoundComponent }
];
@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
